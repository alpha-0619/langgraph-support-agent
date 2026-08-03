"""Graph state.

One dict flows through every node. Nodes return partial updates; LangGraph
merges them. `trace` is the exception: it accumulates instead of overwriting,
which is what makes a finished run auditable after the fact.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Outcome = Literal["sent", "skipped", "duplicate", "escalated", "rejected", "pending"]


class Contact(TypedDict, total=False):
    """What we know about the person on the other end."""

    contact_id: str
    plan: str | None  # "trial" | "paid" | "enterprise" | None
    plan_expires_at: str | None  # ISO 8601
    last_seen_at: str | None  # ISO 8601
    lifetime_value: float
    message_count: int
    tags: list[str]


class SupportState(TypedDict, total=False):
    # --- input ---
    conversation_id: str
    message: str
    contact: Contact
    history: list[tuple[str, str]]  # [(speaker, text), ...] oldest first

    # --- derived ---
    lifecycle: str  # see lifecycle.Stage
    lifecycle_reason: str
    intent: str
    urgency: str
    retrieved: list[str]
    draft: str

    # --- control flow ---
    halt_reason: str | None
    policy_issues: list[str]
    revisions: int
    canned_blocked: bool
    outcome: Outcome

    # --- human-in-the-loop ---
    approval: dict[str, Any] | None  # {"decision": ..., "edited_text": ...}

    # --- observability ---
    trace: Annotated[list[str], operator.add]


def new_state(
    conversation_id: str,
    message: str,
    contact: Contact | None = None,
    history: list[tuple[str, str]] | None = None,
) -> SupportState:
    """Build a fresh state with the control fields zeroed out."""
    return SupportState(
        conversation_id=conversation_id,
        message=message,
        contact=contact or {},
        history=history or [],
        lifecycle="",
        lifecycle_reason="",
        intent="",
        urgency="",
        retrieved=[],
        draft="",
        halt_reason=None,
        policy_issues=[],
        revisions=0,
        canned_blocked=False,
        outcome="pending",
        approval=None,
        trace=[],
    )
