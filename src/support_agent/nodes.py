"""Graph nodes and the dependencies they need.

Every node takes the state and returns a partial update. Nothing here imports
LangGraph except the human-approval node, which needs `interrupt`. That keeps
the business logic testable on its own, without building a graph first.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.types import interrupt

from . import lifecycle, policy
from .llm import Provider, RuleBasedStub, classify_prompt, draft_prompt
from .state import SupportState


class DedupStore:
    """Remembers what we already handled, so a double-send is a no-op.

    Real deployments back this with Redis. The interface is two methods, which
    is the whole reason it is a class and not a module-level dict.
    """

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        self.ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    @staticmethod
    def key(conversation_id: str, message: str) -> str:
        digest = hashlib.sha256(message.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"{conversation_id}:{digest}"

    def seen(self, key: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        stamp = self._seen.get(key)
        if stamp is None:
            return False
        if now - stamp > self.ttl:
            del self._seen[key]
            return False
        return True

    def remember(self, key: str, *, now: float | None = None) -> None:
        self._seen[key] = now if now is not None else time.time()


class CannedLibrary:
    """Pre-approved responses for intents that must not be improvised.

    `blocked_for` is the interesting part. A canned reply can be suppressed
    (already sent recently, campaign paused, contact opted out). When that
    happens the request must NOT vanish: it falls through to the model. Silent
    drops on a blocked fast path are how customers end up ignored for a day.
    """

    def __init__(
        self,
        entries: dict[str, str] | None = None,
        blocked_for: set[str] | None = None,
    ) -> None:
        self.entries = entries or {
            "how_to": "Here is the step-by-step for that, straight from the docs.",
            "cancellation": "I can start the cancellation, and here is what happens to your data.",
        }
        self.blocked_for = blocked_for or set()

    def lookup(self, intent: str, contact_id: str) -> tuple[str | None, bool]:
        """Return (text, blocked). `blocked` distinguishes 'suppressed' from 'no entry'."""
        if intent not in self.entries:
            return None, False
        if contact_id in self.blocked_for:
            return None, True
        return self.entries[intent], False


class KnowledgeBase:
    """Stand-in for the retrieval layer.

    Swapping this for pgvector or Qdrant means implementing `search`. The graph
    does not care, which is the point of keeping retrieval behind one method.
    """

    def __init__(self, docs: dict[str, list[str]] | None = None) -> None:
        self.docs = docs or {
            "billing": [
                "Refunds are processed back to the original payment method.",
                "Billing questions are answered by the finance team within one business day.",
            ],
            "bug_report": [
                "Bug reports need app version, OS, and reproduction steps.",
            ],
            "sales_enquiry": [
                "Pricing is published on the public pricing page and depends on seat count.",
            ],
        }

    def search(self, intent: str, query: str, k: int = 3) -> list[str]:
        return self.docs.get(intent, [])[:k]


@dataclass
class Deps:
    provider: Provider = field(default_factory=RuleBasedStub)
    dedup: DedupStore = field(default_factory=DedupStore)
    canned: CannedLibrary = field(default_factory=CannedLibrary)
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    now: datetime | None = None  # injectable clock, so lifecycle tests are not flaky

    def clock(self) -> datetime:
        return self.now or datetime.now()


# -- nodes -------------------------------------------------------------


def make_nodes(deps: Deps) -> dict:
    def ingest(state: SupportState) -> dict:
        msg = (state.get("message") or "").strip()
        return {"message": msg, "trace": [f"ingest: {len(msg)} chars"]}

    def dedup_gate(state: SupportState) -> dict:
        key = deps.dedup.key(state["conversation_id"], state["message"])
        if deps.dedup.seen(key):
            return {
                "halt_reason": "duplicate message inside the dedup window",
                "outcome": "duplicate",
                "trace": ["dedup: hit, short-circuit"],
            }
        deps.dedup.remember(key)
        return {"trace": ["dedup: miss"]}

    def classify(state: SupportState) -> dict:
        raw = deps.provider.complete(
            classify_prompt(state["message"], state.get("history", [])),
            max_tokens=200,
            temperature=0.0,
        )
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # A malformed classification must not take the run down. Degrade to
            # the catch-all intent and let policy plus the human gate cover us.
            parsed = {"intent": "other", "urgency": "low", "buying": False, "engaged": False}
        stage, reason = lifecycle.determine_stage(
            state.get("contact", {}),
            signals={"buying": parsed.get("buying"), "engaged": parsed.get("engaged")},
            now=deps.clock(),
        )
        return {
            "intent": parsed.get("intent", "other"),
            "urgency": parsed.get("urgency", "low"),
            "lifecycle": stage.value,
            "lifecycle_reason": reason,
            "trace": [f"classify: intent={parsed.get('intent')} stage={stage.value} ({reason})"],
        }

    def triage_gate(state: SupportState) -> dict:
        """Not every message deserves a reply.

        Answering everything trains people to treat the channel as a toy and
        burns tokens on messages that were never a question. Silence is a valid
        outcome, but it has to be a recorded decision, not an accident.
        """
        contact = state.get("contact", {})
        stage = state.get("lifecycle", "")
        intent = state.get("intent", "")
        ltv = float(contact.get("lifetime_value") or 0)

        skip = (
            intent == "smalltalk"
            and stage in (lifecycle.Stage.NEW.value, lifecycle.Stage.DORMANT.value)
            and ltv == 0
        )
        if skip:
            return {
                "halt_reason": "low-signal smalltalk from an unqualified contact",
                "outcome": "skipped",
                "trace": ["triage: skip"],
            }
        return {"trace": ["triage: proceed"]}

    def canned_response(state: SupportState) -> dict:
        contact_id = state.get("contact", {}).get("contact_id", "")
        text, blocked = deps.canned.lookup(state["intent"], contact_id)
        if text:
            return {"draft": text, "trace": ["canned: hit"]}
        if blocked:
            # The edge that matters. Suppressed fast path hands off to the model
            # instead of returning nothing.
            return {
                "canned_blocked": True,
                "trace": ["canned: suppressed, falling through to model"],
            }
        return {"trace": ["canned: no entry, falling through to model"]}

    def retrieve(state: SupportState) -> dict:
        hits = deps.knowledge.search(state["intent"], state["message"])
        return {"retrieved": hits, "trace": [f"retrieve: {len(hits)} passages"]}

    def draft(state: SupportState) -> dict:
        revisions = int(state.get("revisions") or 0)
        text = deps.provider.complete(
            draft_prompt(
                state["message"],
                state.get("intent", "other"),
                state.get("lifecycle", ""),
                state.get("retrieved", []),
                state.get("history", []),
                fix=state.get("policy_issues") or None,
            ),
            max_tokens=400,
            temperature=0.4,
        )
        label = "draft" if revisions == 0 else f"redraft #{revisions}"
        return {
            "draft": text.strip(),
            "revisions": revisions + 1,
            "trace": [f"{label}: {len(text.strip())} chars"],
        }

    def policy_check(state: SupportState) -> dict:
        verdict = policy.check(state.get("draft", ""), state.get("retrieved", []))
        if verdict:
            return {"policy_issues": [], "trace": ["policy: pass"]}
        return {
            "policy_issues": verdict.issues,
            "trace": [f"policy: {len(verdict.issues)} issue(s) - {'; '.join(verdict.issues)}"],
        }

    def escalate(state: SupportState) -> dict:
        return {
            "outcome": "escalated",
            "halt_reason": "policy could not be satisfied within the rewrite budget",
            "trace": ["escalate: handed to a human queue"],
        }

    def approval_gate(state: SupportState) -> dict:
        """Pause the run and wait for a person.

        `interrupt` checkpoints the whole graph. The process can die here and a
        different worker can resume it hours later with the reviewer's decision.
        """
        decision = interrupt(
            {
                "conversation_id": state["conversation_id"],
                "customer_message": state["message"],
                "proposed_reply": state.get("draft", ""),
                "intent": state.get("intent"),
                "lifecycle": state.get("lifecycle"),
                "why": state.get("lifecycle_reason"),
            }
        )
        if isinstance(decision, str):
            decision = {"decision": decision}
        decision = decision or {}
        verb = decision.get("decision", "reject")
        edited = decision.get("edited_text")
        return {
            "approval": decision,
            "draft": edited or state.get("draft", ""),
            "trace": [f"approval: {verb}" + (" (edited)" if edited else "")],
        }

    def send(state: SupportState) -> dict:
        return {"outcome": "sent", "trace": ["send: dispatched"]}

    def reject(state: SupportState) -> dict:
        return {
            "outcome": "rejected",
            "halt_reason": "reviewer rejected the draft",
            "trace": ["reject: discarded by reviewer"],
        }

    return {
        "ingest": ingest,
        "dedup_gate": dedup_gate,
        "classify": classify,
        "triage_gate": triage_gate,
        "canned_response": canned_response,
        "retrieve": retrieve,
        "draft": draft,
        "policy_check": policy_check,
        "escalate": escalate,
        "approval_gate": approval_gate,
        "send": send,
        "reject": reject,
    }
