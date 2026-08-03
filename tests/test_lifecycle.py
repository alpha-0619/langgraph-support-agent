from datetime import datetime, timedelta

import pytest

from support_agent.lifecycle import (
    Stage,
    can_transition,
    determine_stage,
    effort_budget,
    hours_since_last_seen,
)

NOW = datetime(2026, 8, 3, 12, 0, 0)


def contact(**kw):
    base = {
        "contact_id": "c-1",
        "plan": None,
        "plan_expires_at": None,
        "last_seen_at": (NOW - timedelta(minutes=5)).isoformat(),
        "lifetime_value": 0.0,
        "message_count": 0,
    }
    base.update(kw)
    return base


LIVE = (NOW + timedelta(days=30)).isoformat()
EXPIRED = (NOW - timedelta(days=30)).isoformat()
SILENT = (NOW - timedelta(hours=80)).isoformat()


@pytest.mark.parametrize(
    "kwargs,signals,expected",
    [
        # new until there is enough conversation to call it engaged
        ({"message_count": 1}, {}, Stage.NEW),
        ({"message_count": 3}, {"engaged": True}, Stage.NEW),
        ({"message_count": 9}, {"engaged": True}, Stage.ACTIVE),
        # buying signal promotes regardless of message count
        ({"message_count": 1}, {"buying": True}, Stage.INTENT),
        # live plan
        ({"plan": "paid", "plan_expires_at": LIVE}, {}, Stage.CUSTOMER),
        ({"plan": "enterprise", "plan_expires_at": LIVE}, {}, Stage.CUSTOMER),
        # lapsed plan routes by signal instead of assuming a renewal pitch
        ({"plan": "paid", "plan_expires_at": EXPIRED}, {"buying": True}, Stage.INTENT),
        ({"plan": "paid", "plan_expires_at": EXPIRED}, {"engaged": True}, Stage.ACTIVE),
        ({"plan": "paid", "plan_expires_at": EXPIRED}, {}, Stage.NEW),
        # advocate needs both money and a live plan
        (
            {"plan": "paid", "plan_expires_at": LIVE, "lifetime_value": 5000.0},
            {},
            Stage.ADVOCATE,
        ),
        ({"plan": None, "lifetime_value": 5000.0}, {}, Stage.NEW),
    ],
)
def test_stage_table(kwargs, signals, expected):
    stage, _ = determine_stage(contact(**kwargs), signals=signals, now=NOW)
    assert stage is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"last_seen_at": SILENT},
        {"last_seen_at": SILENT, "plan": "paid", "plan_expires_at": LIVE},
        {"last_seen_at": SILENT, "plan": "paid", "plan_expires_at": LIVE, "lifetime_value": 5000.0},
    ],
)
def test_dormancy_outranks_everything_below_it(kwargs):
    """A message after days of silence is a re-engagement, not a hot lead."""
    stage, reason = determine_stage(
        contact(**kwargs), signals={"buying": True, "engaged": True}, now=NOW
    )
    assert stage is Stage.DORMANT
    assert "idle" in reason or "since last contact" in reason


def test_unknown_last_seen_does_not_mark_dormant():
    stage, _ = determine_stage(contact(last_seen_at=None, message_count=1), now=NOW)
    assert stage is Stage.NEW


def test_malformed_timestamp_is_ignored_not_fatal():
    assert hours_since_last_seen(contact(last_seen_at="not-a-date"), NOW) is None
    stage, _ = determine_stage(contact(last_seen_at="not-a-date", message_count=1), now=NOW)
    assert stage is Stage.NEW


def test_transitions_are_bounded():
    assert can_transition(Stage.NEW, Stage.ACTIVE)
    assert can_transition(Stage.CUSTOMER, Stage.ADVOCATE)
    assert can_transition(Stage.ADVOCATE, Stage.ADVOCATE)  # idempotent
    # an advocate must not be dragged back into onboarding by one bad read
    assert not can_transition(Stage.ADVOCATE, Stage.NEW)
    assert not can_transition(Stage.CUSTOMER, Stage.NEW)


def test_dormant_can_revive_to_any_stage():
    for target in Stage:
        if target is Stage.DORMANT:
            continue
        assert can_transition(Stage.DORMANT, target)


def test_effort_budget_favours_paying_contacts():
    assert effort_budget(Stage.CUSTOMER) > effort_budget(Stage.NEW)
    assert effort_budget(Stage.INTENT) > effort_budget(Stage.DORMANT)
    assert all(effort_budget(s) >= 1 for s in Stage)
