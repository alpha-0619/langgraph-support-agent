"""Runnable tour of every path through the graph.

No API key and no network: the default provider is the deterministic stub, so
this prints the same thing on any machine.

    python demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

from support_agent import CannedLibrary, DedupStore, Deps, build_graph, new_state  # noqa: E402
from support_agent.llm import ScriptedStub  # noqa: E402
from langgraph.types import Command  # noqa: E402

BILLING_SIGNAL = '{"intent": "billing", "urgency": "low", "buying": false, "engaged": true}'

NOW = datetime(2026, 8, 3, 18, 0, 0)
RECENT = (NOW - timedelta(minutes=20)).isoformat()
STALE = (NOW - timedelta(days=9)).isoformat()

RULE = "=" * 72


def show(title: str, state: dict) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")
    for line in state.get("trace", []):
        print(f"  . {line}")
    print(f"  outcome : {state.get('outcome')}")
    if state.get("halt_reason"):
        print(f"  reason  : {state['halt_reason']}")
    if state.get("draft"):
        print(f"  reply   : {state['draft']}")


def run(graph, state, thread: str, *, resume=None):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke(state, config)
    if "__interrupt__" in result and resume is not None:
        payload = result["__interrupt__"][0].value
        print(f"\n  [paused for review] proposed: {payload['proposed_reply']!r}")
        result = graph.invoke(Command(resume=resume), config)
    return result


def main() -> None:
    paying = {
        "contact_id": "c-100",
        "plan": "paid",
        "plan_expires_at": (NOW + timedelta(days=40)).isoformat(),
        "last_seen_at": RECENT,
        "lifetime_value": 240.0,
        "message_count": 31,
    }
    stranger = {
        "contact_id": "c-200",
        "plan": None,
        "last_seen_at": RECENT,
        "lifetime_value": 0.0,
        "message_count": 1,
    }

    # 1. happy path, reviewer approves
    graph = build_graph(Deps(now=NOW))
    show(
        "1. Paying customer, billing question, reviewer approves",
        run(
            graph,
            new_state("conv-1", "I was charged twice for my invoice this month", paying),
            "t1",
            resume={"decision": "approve"},
        ),
    )

    # 2. same text again inside the dedup window
    show(
        "2. Same message resent, deduped before any model call",
        run(graph, new_state("conv-1", "I was charged twice for my invoice this month", paying), "t2"),
    )

    # 3. triage skip
    show(
        "3. Cold contact says 'hi', triaged out",
        run(graph, new_state("conv-2", "hi", stranger), "t3"),
    )

    # 4. suppressed canned reply falls through to the model
    suppressed = Deps(
        now=NOW,
        dedup=DedupStore(),
        canned=CannedLibrary(blocked_for={"c-100"}),
    )
    show(
        "4. Canned reply suppressed for this contact, falls through instead of dropping",
        run(
            build_graph(suppressed),
            new_state("conv-3", "How do I export my data?", paying),
            "t4",
            resume={"decision": "approve"},
        ),
    )

    # 5. reviewer edits before sending
    show(
        "5. Reviewer edits the draft, edited text is what ships",
        run(
            build_graph(Deps(now=NOW)),
            new_state("conv-4", "the app crashes on launch, this is urgent", paying),
            "t5",
            resume={"decision": "approve_with_edit", "edited_text": "Rolling back the 4.2 build now."},
        ),
    )

    # 6. reviewer rejects
    show(
        "6. Reviewer rejects, nothing is sent",
        run(
            build_graph(Deps(now=NOW)),
            new_state("conv-5", "the app crashes on launch", paying),
            "t6",
            resume={"decision": "reject"},
        ),
    )

    # 7. dormant contact, stage overrides the buying signal
    dormant = dict(paying, contact_id="c-300", last_seen_at=STALE)
    show(
        "7. Nine days silent, dormancy outranks the buying signal",
        run(
            build_graph(Deps(now=NOW)),
            new_state("conv-6", "what is your enterprise pricing?", dormant),
            "t7",
            resume={"decision": "approve"},
        ),
    )

    # 8. first draft breaks policy, the loop rewrites it, second one ships
    recovers = Deps(
        now=NOW,
        provider=ScriptedStub(
            classify=BILLING_SIGNAL,
            drafts=[
                "We guarantee a full refund of $250 within 3 business days.",
                "I have asked finance to look at the duplicate charge and they will confirm it.",
            ],
        ),
    )
    show(
        "8. Draft invents a figure and promises a refund, rewrite loop catches it",
        run(
            build_graph(recovers),
            new_state("conv-7", "I was double charged, when do I get my money back?", paying),
            "t8",
            resume={"decision": "approve"},
        ),
    )

    # 9. same mistake twice, budget runs out, nothing reaches the customer
    stubborn = Deps(
        now=NOW,
        provider=ScriptedStub(
            classify=BILLING_SIGNAL,
            drafts=["We guarantee a refund of $500 today.", "I promise you a refund of $500 today."],
        ),
    )
    show(
        "9. Model will not comply, rewrite budget spent, escalated instead of sent",
        run(
            build_graph(stubborn),
            new_state("conv-8", "I was double charged, when do I get my money back?", paying),
            "t9",
        ),
    )


if __name__ == "__main__":
    main()
