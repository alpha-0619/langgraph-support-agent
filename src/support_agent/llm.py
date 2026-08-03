"""Model access, behind a narrow interface.

The graph never imports an SDK directly, for two reasons. `RuleBasedStub` makes
every branch reachable without a network call or an API key, which keeps the
suite runnable anywhere. And changing provider stays a config change.

The stub is deliberately crude. It is a fixture for exercising control flow, not
an attempt to approximate a model.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

CATEGORIES = [
    "billing",
    "bug_report",
    "how_to",
    "feature_request",
    "cancellation",
    "sales_enquiry",
    "complaint",
    "smalltalk",
    "other",
]


class Provider(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.3) -> str: ...


class RuleBasedStub:
    """Deterministic stand-in. Same interface, no network."""

    name = "stub"

    _INTENT_PATTERNS: list[tuple[str, str]] = [
        (r"refund|invoice|charge|billing|overcharg|receipt", "billing"),
        (r"crash|error|broken|not work|bug|fails|500|timeout", "bug_report"),
        (r"cancel|unsubscrib|downgrade|close my account", "cancellation"),
        (r"pricing|quote|demo|enterprise plan|how much|buy|upgrade", "sales_enquiry"),
        (r"how do i|how to|where is|can i|is it possible", "how_to"),
        (r"would be great if|feature|please add|wish", "feature_request"),
        (r"terrible|awful|unacceptable|furious|worst", "complaint"),
        (r"^(hi|hello|hey|thanks|thank you|ok|okay)\b", "smalltalk"),
    ]

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if "<<TASK:CLASSIFY>>" in prompt:
            return self._classify(prompt)
        if "<<TASK:DRAFT>>" in prompt:
            return self._draft(prompt)
        return ""

    # -- task handlers -------------------------------------------------

    def _classify(self, prompt: str) -> str:
        msg = _extract(prompt, "MESSAGE").lower()
        intent = "other"
        for pattern, label in self._INTENT_PATTERNS:
            if re.search(pattern, msg):
                intent = label
                break
        urgency = "high" if re.search(r"urgent|asap|immediately|down|outage|furious", msg) else "low"
        buying = bool(re.search(r"pricing|quote|demo|how much|buy|upgrade|enterprise", msg))
        engaged = len(msg.split()) > 6
        return json.dumps(
            {"intent": intent, "urgency": urgency, "buying": buying, "engaged": engaged}
        )

    def _draft(self, prompt: str) -> str:
        intent = _extract(prompt, "INTENT")
        issues = _extract(prompt, "FIX")
        base = {
            "billing": "I pulled up the charge on your account and I can see what happened.",
            "bug_report": "Thanks for the detail, that is enough for me to reproduce it.",
            "how_to": "Export lives under Account, and I will point you at the exact screen.",
            "cancellation": "I can process that for you, and I want to check one thing first.",
            "sales_enquiry": "Happy to talk through pricing and what fits your volume.",
            "complaint": "That is a fair thing to be annoyed about, and I want to fix it.",
            "feature_request": "Logged, and I will tell you honestly where it sits.",
            "smalltalk": "Hello, what can I help with?",
        }.get(intent, "Let me take a look at that for you.")
        if issues:
            # Second pass: the policy checker told us what was wrong.
            base = base.replace("!", ".")
            base = re.sub(r"\b(guarantee|guaranteed|always|never|promise)\b", "expect", base)
            base += " I will confirm the specifics before anything is final."
        return base


class ScriptedStub:
    """Plays back a fixed list of completions, in order.

    For exercising paths the rule-based stub will never produce on its own:
    a draft that breaks policy, a malformed classification, a model that keeps
    making the same mistake until the rewrite budget runs out. The last entry
    repeats once the script is exhausted.
    """

    name = "scripted"

    def __init__(self, classify: str | None = None, drafts: list[str] | None = None) -> None:
        self._classify = classify or RuleBasedStub().complete("<<TASK:CLASSIFY>>[MESSAGE][/MESSAGE]")
        self._drafts = list(drafts or [""])
        self._i = 0

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if "<<TASK:CLASSIFY>>" in prompt:
            return self._classify
        text = self._drafts[min(self._i, len(self._drafts) - 1)]
        self._i += 1
        return text


class AnthropicProvider:
    """Real provider. Imported lazily so the base install stays dependency-light."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = None

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if self._client is None:
            from anthropic import Anthropic  # noqa: PLC0415 - deliberate lazy import

            self._client = Anthropic(api_key=self._api_key)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_reasoning(text: str) -> str:
    """Drop the reasoning block some models emit before their answer.

    An unclosed block means the model spent its whole budget thinking and never
    produced an answer. Returning the reasoning would put a monologue in front
    of a customer, so that case returns empty and the policy layer treats it as
    a failed draft.
    """
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE, 1)[1].strip()
    if _THINK_OPEN in text:
        return ""
    return text.strip()


class MiniMaxProvider:
    """MiniMax M2 over its OpenAI-compatible endpoint.

    M2 is a reasoning model, so `max_tokens` carries an extra allowance for the
    reasoning block on top of what the caller asked for. Without it the answer
    gets truncated away and every draft comes back empty.
    """

    name = "minimax"
    REASONING_ALLOWANCE = 800

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model or os.environ.get("MINIMAX_MODEL") or "MiniMax-M2"
        self.base_url = base_url or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimax.io/v1"
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self._api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        self._client = None

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.3) -> str:
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415 - deliberate lazy import

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a customer support agent. Follow the task markers in the user message exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens + self.REASONING_ALLOWANCE,
        )
        return strip_reasoning(resp.choices[0].message.content or "")


def get_provider(name: str | None = None) -> Provider:
    """Resolve a provider by name, defaulting to the offline stub."""
    name = (name or os.environ.get("SUPPORT_AGENT_PROVIDER") or "stub").lower()
    if name == "stub":
        return RuleBasedStub()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "minimax":
        return MiniMaxProvider()
    raise ValueError(f"unknown provider: {name!r}")


# -- prompt helpers ----------------------------------------------------

_FIELD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _extract(prompt: str, field: str) -> str:
    """Read back a delimited field the stub itself was handed."""
    pattern = _FIELD_RE_CACHE.get(field)
    if pattern is None:
        pattern = re.compile(rf"\[{field}\](.*?)\[/{field}\]", re.S)
        _FIELD_RE_CACHE[field] = pattern
    match = pattern.search(prompt)
    return match.group(1).strip() if match else ""


# The prompt format uses bracketed field markers. A customer who writes one of
# them into their message can close a field early and open a fake one, so the
# model reads attacker text as if it were an instruction from us. Untrusted
# text gets these tokens defanged on the way in.
_DELIMITERS = re.compile(
    r"\[/?(?:MESSAGE|HISTORY|KNOWLEDGE|INTENT|STAGE|FIX)\]|<<\s*TASK\s*:[^>]*>>",
    re.I,
)


def neutralise(text: str) -> str:
    """Strip the control tokens this prompt format relies on."""
    return _DELIMITERS.sub("_", text or "")


def classify_prompt(message: str, history: list[tuple[str, str]]) -> str:
    message = neutralise(message)
    history = [(who, neutralise(text)) for who, text in history]
    recent = "\n".join(f"{who}: {text[:160]}" for who, text in history[-5:]) or "(none)"
    return (
        "<<TASK:CLASSIFY>>\n"
        "Classify the customer message. Reply with JSON only, no prose.\n"
        f"Allowed intents: {', '.join(CATEGORIES)}\n"
        'Schema: {"intent": str, "urgency": "low"|"high", "buying": bool, "engaged": bool}\n'
        f"[HISTORY]\n{recent}\n[/HISTORY]\n"
        f"[MESSAGE]\n{message}\n[/MESSAGE]"
    )


def draft_prompt(
    message: str,
    intent: str,
    stage: str,
    knowledge: list[str],
    history: list[tuple[str, str]],
    fix: list[str] | None = None,
) -> str:
    message = neutralise(message)
    history = [(who, neutralise(text)) for who, text in history]
    recent = "\n".join(f"{who}: {text[:160]}" for who, text in history[-5:]) or "(none)"
    kb = "\n".join(f"- {neutralise(k)}" for k in knowledge) or "(no retrieved context)"
    fix_block = ""
    if fix:
        fix_block = "[FIX]\n" + "\n".join(f"- {f}" for f in fix) + "\n[/FIX]\n"
    return (
        "<<TASK:DRAFT>>\n"
        "Write one reply to the customer. Plain text, no greeting boilerplate.\n"
        "Ground every factual claim in the retrieved context. If the context does "
        "not cover it, say you will check rather than inventing a specific.\n"
        f"[INTENT]{intent}[/INTENT]\n"
        f"[STAGE]{stage}[/STAGE]\n"
        f"{fix_block}"
        f"[KNOWLEDGE]\n{kb}\n[/KNOWLEDGE]\n"
        f"[HISTORY]\n{recent}\n[/HISTORY]\n"
        f"[MESSAGE]\n{message}\n[/MESSAGE]"
    )
