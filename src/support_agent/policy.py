"""Deterministic policy checks on generated text.

These are rules, not a model call, and that is the point. An LLM asked to
police another LLM gives you a different answer on Tuesday. Compliance rules
that decide whether something reaches a customer need to be reproducible and
auditable, so they live here as code with a test each.

The model handles language. Rules handle whether it is allowed out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Commitments nobody is authorised to make in an automated reply.
_PROMISES = re.compile(
    r"\b(guarantee|guaranteed|we promise|i promise|always will|never fails?|"
    r"100% (?:safe|secure|uptime)|refund(?:ed)? (?:today|immediately))\b",
    re.I,
)

# Invented specifics. A support agent quoting a number it did not retrieve is
# the single most expensive failure mode in this kind of system.
_UNGROUNDED_NUMBER = re.compile(
    r"(?<![\w.])(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*\s?(?:%|percent|days?|hours?|business days?))",
    re.I,
)

_LEGAL_OR_MEDICAL = re.compile(r"\b(legally|liable|lawsuit|sue|diagnos\w+|prescri\w+)\b", re.I)

_PII_ECHO = re.compile(
    r"\b(?:\d[ -]?){13,16}\b"  # card-shaped
    r"|\b[\w.+-]+@[\w-]+\.[\w.]+\b"  # email
    r"|\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?){2,4}\d{2,4}\b",  # phone-shaped
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    issues: list[str]

    def __bool__(self) -> bool:  # `if verdict:` reads naturally
        return self.ok


def _normalise(token: str) -> str:
    """Fold spacing so '$ 250' and '$250' compare equal."""
    return re.sub(r"\s+", "", token.strip().lower())


def check(draft: str, knowledge: list[str] | None = None) -> Verdict:
    """Return every rule the draft breaks, not just the first."""
    knowledge = knowledge or []
    issues: list[str] = []

    if not draft.strip():
        return Verdict(False, ["empty draft"])

    if _PROMISES.search(draft):
        issues.append("makes a commitment the agent cannot authorise")

    # Compare whole extracted quantities, never substrings. A substring test
    # passes "$25" against a context containing "$250", which is exactly the
    # ungrounded figure this rule exists to stop.
    grounded = {_normalise(t) for t in _UNGROUNDED_NUMBER.findall(" ".join(knowledge))}
    for number in _UNGROUNDED_NUMBER.findall(draft):
        token = number.strip()
        if token and _normalise(token) not in grounded:
            issues.append(f"states {token!r}, which is not in the retrieved context")

    if _LEGAL_OR_MEDICAL.search(draft):
        issues.append("strays into legal or medical advice")

    if _PII_ECHO.search(draft):
        issues.append("echoes personally identifying data back to the channel")

    if len(draft) > 1200:
        issues.append("too long for a first reply")

    # Dedupe while keeping order, so the rewrite prompt is not repetitive.
    seen: set[str] = set()
    ordered = [i for i in issues if not (i in seen or seen.add(i))]
    return Verdict(not ordered, ordered)
