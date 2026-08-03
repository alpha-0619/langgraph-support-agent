"""Graph wiring.

    START
      -> ingest
      -> dedup_gate ------------(duplicate)-------------> END
      -> classify  (intent + lifecycle stage in one pass)
      -> triage_gate -----------(not worth a reply)-----> END
      -> canned_response
           |-- hit -----------------------------> policy_check
           `-- suppressed / no entry -----------> retrieve -> draft -> policy_check
      -> policy_check
           |-- pass ----------------------------> approval_gate
           |-- fixable, budget left ------------> retrieve -> draft   (loop)
           `-- budget spent --------------------> escalate -> END
      -> approval_gate  [interrupt: waits for a human]
           |-- approve -------------------------> send   -> END
           `-- reject --------------------------> reject -> END

Three things here are the reason this is a graph and not a chain: the two
gates that can end the run before any model call, the fallback edge out of the
suppressed fast path, and the bounded rewrite loop.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import lifecycle
from .nodes import Deps, make_nodes
from .state import SupportState


def _stage_of(state: SupportState) -> lifecycle.Stage:
    try:
        return lifecycle.Stage(state.get("lifecycle") or "")
    except ValueError:
        return lifecycle.Stage.NEW


def route_after_dedup(state: SupportState) -> str:
    return "halt" if state.get("halt_reason") else "continue"


def route_after_triage(state: SupportState) -> str:
    return "halt" if state.get("halt_reason") else "continue"


def route_after_canned(state: SupportState) -> str:
    return "check" if state.get("draft") else "generate"


def route_after_policy(state: SupportState) -> str:
    if not state.get("policy_issues"):
        return "approve"
    budget = lifecycle.effort_budget(_stage_of(state))
    if int(state.get("revisions") or 0) < budget:
        return "revise"
    return "escalate"


def route_after_approval(state: SupportState) -> str:
    decision = (state.get("approval") or {}).get("decision")
    return "send" if decision in ("approve", "approve_with_edit") else "reject"


def build_graph(deps: Deps | None = None, *, checkpointer=None):
    """Compile the graph.

    A checkpointer is required for the human approval step: `interrupt` has to
    persist state somewhere before it can be resumed. In-memory is fine for
    tests and demos; production swaps in Postgres and nothing else changes.
    """
    deps = deps or Deps()
    nodes = make_nodes(deps)

    builder = StateGraph(SupportState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "dedup_gate")
    builder.add_conditional_edges(
        "dedup_gate", route_after_dedup, {"halt": END, "continue": "classify"}
    )
    builder.add_edge("classify", "triage_gate")
    builder.add_conditional_edges(
        "triage_gate", route_after_triage, {"halt": END, "continue": "canned_response"}
    )
    builder.add_conditional_edges(
        "canned_response", route_after_canned, {"check": "policy_check", "generate": "retrieve"}
    )
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "policy_check")
    builder.add_conditional_edges(
        "policy_check",
        route_after_policy,
        {"approve": "approval_gate", "revise": "retrieve", "escalate": "escalate"},
    )
    builder.add_conditional_edges(
        "approval_gate", route_after_approval, {"send": "send", "reject": "reject"}
    )
    builder.add_edge("send", END)
    builder.add_edge("reject", END)
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
