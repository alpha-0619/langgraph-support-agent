"""Contact lifecycle state machine.

Six stages, evaluated in priority order, computed before any model call so the
stage can decide how much effort the message is worth.

The shape is taken from a production messaging system where six equivalent
stages drive routing across nine concurrent agent instances.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .state import Contact


class Stage(str, Enum):
    NEW = "new"  # first contact, nothing known yet
    ACTIVE = "active"  # engaged, no purchase
    INTENT = "intent"  # showing buying / upgrade signals
    CUSTOMER = "customer"  # paying, subscription live
    ADVOCATE = "advocate"  # long-term, high lifetime value
    DORMANT = "dormant"  # silent past the threshold


# Which transitions are legal. Enforced so a bad signal read cannot teleport a
# paying customer back to NEW and re-trigger onboarding at them.
VALID_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.NEW: {Stage.ACTIVE, Stage.INTENT, Stage.DORMANT},
    Stage.ACTIVE: {Stage.INTENT, Stage.NEW, Stage.DORMANT},
    Stage.INTENT: {Stage.CUSTOMER, Stage.ACTIVE, Stage.DORMANT},
    Stage.CUSTOMER: {Stage.ADVOCATE, Stage.INTENT, Stage.DORMANT},
    Stage.ADVOCATE: {Stage.DORMANT},
    Stage.DORMANT: set(Stage) - {Stage.DORMANT},  # revival returns to any stage
}

DORMANT_AFTER_HOURS = 72.0
ACTIVE_AFTER_MESSAGES = 5
ADVOCATE_LTV = 1000.0


def can_transition(src: Stage, dst: Stage) -> bool:
    if src == dst:
        return True  # idempotent
    return dst in VALID_TRANSITIONS.get(src, set())


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def hours_since_last_seen(contact: Contact, now: datetime) -> float | None:
    seen = _parse_iso(contact.get("last_seen_at"))
    if seen is None:
        return None
    return (_naive(now) - _naive(seen)).total_seconds() / 3600


def plan_is_live(contact: Contact, now: datetime) -> bool:
    expires = _parse_iso(contact.get("plan_expires_at"))
    if expires is None:
        return False
    return _naive(expires) > _naive(now)


def determine_stage(
    contact: Contact,
    signals: dict | None = None,
    now: datetime | None = None,
) -> tuple[Stage, str]:
    """Return (stage, human-readable reason).

    Order is deliberate. Dormancy is checked ahead of buying signals because a
    message after three days of silence is a re-engagement first.
    """
    now = now or datetime.now()
    signals = signals or {}

    plan = contact.get("plan")
    ltv = float(contact.get("lifetime_value") or 0)
    msgs = int(contact.get("message_count") or 0)
    idle = hours_since_last_seen(contact, now)
    live = plan_is_live(contact, now)
    buying = bool(signals.get("buying"))
    engaged = bool(signals.get("engaged"))

    # 1. advocate, unless they have gone quiet
    if ltv >= ADVOCATE_LTV and plan in ("paid", "enterprise"):
        if idle is not None and idle >= DORMANT_AFTER_HOURS:
            return Stage.DORMANT, f"advocate idle {idle:.1f}h"
        return Stage.ADVOCATE, f"lifetime value {ltv:.0f} on {plan}"

    # 2. dormant beats everything below it
    if idle is not None and idle >= DORMANT_AFTER_HOURS:
        return Stage.DORMANT, f"{idle:.1f}h since last contact (>= {DORMANT_AFTER_HOURS:.0f}h)"

    # 3. paying customer with a live plan
    if plan in ("paid", "enterprise") and live:
        return Stage.CUSTOMER, f"plan={plan}, active"

    # 4. lapsed plan: route by signal so renewal talk is earned, not sprayed
    if plan in ("paid", "enterprise") and not live:
        if buying:
            return Stage.INTENT, "plan lapsed, buying signal present"
        if engaged:
            return Stage.ACTIVE, "plan lapsed, still engaged"
        return Stage.NEW, "plan lapsed, no signal, re-onboard"

    # 5. buying signal from a non-customer
    if buying:
        return Stage.INTENT, "buying signal"

    # 6. engaged after enough back-and-forth
    if engaged and msgs >= ACTIVE_AFTER_MESSAGES:
        return Stage.ACTIVE, f"engaged over {msgs} messages"

    if msgs < ACTIVE_AFTER_MESSAGES:
        return Stage.NEW, f"only {msgs} messages so far"
    return Stage.ACTIVE, "default"


def effort_budget(stage: Stage) -> int:
    """How many policy-rewrite attempts this stage is worth.

    One means a draft that breaks policy is escalated rather than retried, which
    is the right trade for contacts with nothing at stake yet.
    """
    return {
        Stage.NEW: 1,
        Stage.ACTIVE: 1,
        Stage.INTENT: 2,
        Stage.CUSTOMER: 2,
        Stage.ADVOCATE: 2,
        Stage.DORMANT: 1,
    }.get(stage, 1)
