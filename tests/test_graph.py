"""Path coverage for the graph itself.

Each test pins one route through the state machine. The stub provider makes
these deterministic, so a failure here means the wiring changed, not that a
model had an off day.
"""

from datetime import datetime, timedelta

import pytest
from langgraph.types import Command

from support_agent import CannedLibrary, DedupStore, Deps, build_graph, new_state
from support_agent.llm import ScriptedStub

NOW = datetime(2026, 8, 3, 12, 0, 0)
BILLING = '{"intent": "billing", "urgency": "low", "buying": false, "engaged": true}'


@pytest.fixture
def paying():
    return {
        "contact_id": "c-1",
        "plan": "paid",
        "plan_expires_at": (NOW + timedelta(days=30)).isoformat(),
        "last_seen_at": (NOW - timedelta(minutes=5)).isoformat(),
        "lifetime_value": 240.0,
        "message_count": 30,
    }


@pytest.fixture
def stranger():
    return {
        "contact_id": "c-2",
        "plan": None,
        "last_seen_at": (NOW - timedelta(minutes=5)).isoformat(),
        "lifetime_value": 0.0,
        "message_count": 1,
    }


def drive(graph, state, thread, resume=None):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke(state, config)
    if "__interrupt__" in result and resume is not None:
        result = graph.invoke(Command(resume=resume), config)
    return result


def test_happy_path_reaches_send(paying):
    graph = build_graph(Deps(now=NOW))
    out = drive(
        graph,
        new_state("c", "I was charged twice for my invoice", paying),
        "t",
        {"decision": "approve"},
    )
    assert out["outcome"] == "sent"
    assert out["draft"]
    assert "policy: pass" in out["trace"]


def test_nothing_is_sent_without_a_human_decision(paying):
    """The run must park at the approval gate, not sail through it."""
    graph = build_graph(Deps(now=NOW))
    out = graph.invoke(
        new_state("c", "I was charged twice for my invoice", paying),
        {"configurable": {"thread_id": "t"}},
    )
    assert "__interrupt__" in out
    assert out.get("outcome") != "sent"


def test_interrupt_payload_shows_the_reviewer_what_they_are_approving(paying):
    graph = build_graph(Deps(now=NOW))
    out = graph.invoke(
        new_state("c", "I was charged twice for my invoice", paying),
        {"configurable": {"thread_id": "t"}},
    )
    payload = out["__interrupt__"][0].value
    assert payload["proposed_reply"]
    assert payload["customer_message"] == "I was charged twice for my invoice"
    assert payload["lifecycle"] == "customer"


def test_duplicate_short_circuits_before_any_model_call(paying):
    """A repeat inside the window must not reach classification."""
    graph = build_graph(Deps(now=NOW))
    text = "I was charged twice for my invoice"
    drive(graph, new_state("c", text, paying), "t1", {"decision": "approve"})
    out = drive(graph, new_state("c", text, paying), "t2")
    assert out["outcome"] == "duplicate"
    assert not any("classify" in line for line in out["trace"])


def test_dedup_is_scoped_per_conversation(paying):
    graph = build_graph(Deps(now=NOW))
    text = "I was charged twice for my invoice"
    drive(graph, new_state("conv-a", text, paying), "t1", {"decision": "approve"})
    out = drive(graph, new_state("conv-b", text, paying), "t2", {"decision": "approve"})
    assert out["outcome"] == "sent"


def test_expired_dedup_entry_lets_the_message_through(paying):
    store = DedupStore(ttl_seconds=0.0)
    graph = build_graph(Deps(now=NOW, dedup=store))
    text = "I was charged twice for my invoice"
    drive(graph, new_state("c", text, paying), "t1", {"decision": "approve"})
    out = drive(graph, new_state("c", text, paying), "t2", {"decision": "approve"})
    assert out["outcome"] == "sent"


def test_low_signal_stranger_is_triaged_out(stranger):
    out = drive(build_graph(Deps(now=NOW)), new_state("c", "hi", stranger), "t")
    assert out["outcome"] == "skipped"
    assert not out["draft"]


def test_paying_customer_smalltalk_is_not_triaged_out(paying):
    """The skip rule keys on value, not on the word 'hi'."""
    out = drive(build_graph(Deps(now=NOW)), new_state("c", "hi", paying), "t", {"decision": "approve"})
    assert out["outcome"] == "sent"


def test_canned_hit_skips_generation(paying):
    out = drive(
        build_graph(Deps(now=NOW)),
        new_state("c", "How do I export my data?", paying),
        "t",
        {"decision": "approve"},
    )
    assert "canned: hit" in out["trace"]
    assert not any("retrieve" in line for line in out["trace"])


def test_suppressed_canned_reply_falls_through_instead_of_vanishing(paying):
    """The failure this guards against is a silent drop, not a wrong answer."""
    deps = Deps(now=NOW, canned=CannedLibrary(blocked_for={"c-1"}))
    out = drive(
        build_graph(deps),
        new_state("c", "How do I export my data?", paying),
        "t",
        {"decision": "approve"},
    )
    assert out["outcome"] == "sent"
    assert out["draft"]
    assert any("suppressed" in line for line in out["trace"])


def test_policy_failure_triggers_one_rewrite_then_ships(paying):
    deps = Deps(
        now=NOW,
        provider=ScriptedStub(
            classify=BILLING,
            drafts=[
                "We guarantee a full refund of $250 within 3 business days.",
                "Finance is looking at the duplicate charge and will confirm.",
            ],
        ),
    )
    out = drive(build_graph(deps), new_state("c", "double charged", paying), "t", {"decision": "approve"})
    assert out["outcome"] == "sent"
    assert out["revisions"] == 2
    assert "$250" not in out["draft"]


def test_rewrite_budget_is_bounded_and_ends_in_escalation(paying):
    deps = Deps(
        now=NOW,
        provider=ScriptedStub(classify=BILLING, drafts=["We guarantee a refund of $500 today."]),
    )
    out = drive(build_graph(deps), new_state("c", "double charged", paying), "t")
    assert out["outcome"] == "escalated"
    assert out["revisions"] == 2  # effort_budget(CUSTOMER)
    assert "__interrupt__" not in out  # never offered to a reviewer


def test_reviewer_edit_is_what_ships(paying):
    out = drive(
        build_graph(Deps(now=NOW)),
        new_state("c", "the app crashes on launch", paying),
        "t",
        {"decision": "approve_with_edit", "edited_text": "Rolling back the 4.2 build now."},
    )
    assert out["outcome"] == "sent"
    assert out["draft"] == "Rolling back the 4.2 build now."


def test_reviewer_rejection_sends_nothing(paying):
    out = drive(
        build_graph(Deps(now=NOW)),
        new_state("c", "the app crashes on launch", paying),
        "t",
        {"decision": "reject"},
    )
    assert out["outcome"] == "rejected"


def test_unrecognised_decision_defaults_to_not_sending(paying):
    """Anything that is not an explicit approval must fail closed."""
    out = drive(
        build_graph(Deps(now=NOW)),
        new_state("c", "the app crashes on launch", paying),
        "t",
        {"decision": "maybe later"},
    )
    assert out["outcome"] == "rejected"


def test_malformed_classification_does_not_kill_the_run(paying):
    deps = Deps(
        now=NOW,
        provider=ScriptedStub(classify="not json at all", drafts=["Let me take a look at that."]),
    )
    out = drive(build_graph(deps), new_state("c", "something odd", paying), "t", {"decision": "approve"})
    assert out["outcome"] == "sent"
    assert out["intent"] == "other"


def test_trace_records_every_step_for_audit(paying):
    out = drive(
        build_graph(Deps(now=NOW)),
        new_state("c", "I was charged twice for my invoice", paying),
        "t",
        {"decision": "approve"},
    )
    joined = " | ".join(out["trace"])
    for expected in ["ingest", "dedup", "classify", "triage", "retrieve", "draft", "policy", "approval", "send"]:
        assert expected in joined
