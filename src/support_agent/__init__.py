"""A support-triage agent built as an explicit LangGraph state graph."""

from .graph import build_graph
from .nodes import CannedLibrary, DedupStore, Deps, KnowledgeBase
from .state import Contact, SupportState, new_state

__all__ = [
    "build_graph",
    "Deps",
    "DedupStore",
    "CannedLibrary",
    "KnowledgeBase",
    "SupportState",
    "Contact",
    "new_state",
]
__version__ = "0.1.0"
