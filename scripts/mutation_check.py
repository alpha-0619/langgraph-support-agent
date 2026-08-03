"""Mutation check: break one thing on purpose, confirm the suite notices.

A green suite proves nothing until you have watched it go red for the right
reason. Each mutation below removes one specific safety property and names the
test that must catch it. If a mutation survives, that test is decorative.

    python scripts/mutation_check.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = pathlib.Path(sys.executable)

MUTATIONS = [
    (
        "fallback edge removed: suppressed canned reply dead-ends",
        "src/support_agent/graph.py",
        '{"check": "policy_check", "generate": "retrieve"}',
        '{"check": "policy_check", "generate": END}',
        "test_suppressed_canned_reply_falls_through_instead_of_vanishing",
    ),
    (
        "approval fails open: unknown decision is treated as approval",
        "src/support_agent/graph.py",
        'return "send" if decision in ("approve", "approve_with_edit") else "reject"',
        'return "reject" if decision == "reject" else "send"',
        "test_unrecognised_decision_defaults_to_not_sending",
    ),
    (
        "rewrite loop unbounded: budget check removed",
        "src/support_agent/graph.py",
        'if int(state.get("revisions") or 0) < budget:',
        "if True:",
        "test_rewrite_budget_is_bounded_and_ends_in_escalation",
    ),
    (
        "dormancy no longer outranks buying signals",
        "src/support_agent/lifecycle.py",
        "    if idle is not None and idle >= DORMANT_AFTER_HOURS:\n"
        '        return Stage.DORMANT, f"{idle:.1f}h since last contact '
        '(>= {DORMANT_AFTER_HOURS:.0f}h)"',
        "    if False:\n        pass",
        "test_dormancy_outranks_everything_below_it",
    ),
    (
        "grounding back to a substring test: '$25' passes against '$250'",
        "src/support_agent/policy.py",
        'grounded = {_normalise(t) for t in _UNGROUNDED_NUMBER.findall(" ".join(knowledge))}\n'
        "    for number in _UNGROUNDED_NUMBER.findall(draft):\n"
        "        token = number.strip()\n"
        "        if token and _normalise(token) not in grounded:",
        'grounded = " ".join(knowledge)\n'
        "    for number in _UNGROUNDED_NUMBER.findall(draft):\n"
        "        token = number.strip()\n"
        "        if token and token not in grounded:",
        "test_prefix_of_a_grounded_number_is_still_ungrounded",
    ),
    (
        "prompt field markers no longer stripped from untrusted text",
        "src/support_agent/llm.py",
        'return _DELIMITERS.sub("_", text or "")',
        'return text or ""',
        "test_delimiters_are_stripped_from_the_message",
    ),
]


def run_suite() -> tuple[bool, str]:
    proc = subprocess.run(
        # No -x: we need every failure, not the first one, to tell whether the
        # named guard test is the one that fired.
        [str(PY), "-m", "pytest", "-q", "--no-header", "--tb=no", "-rf"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr)


def main() -> int:
    ok, _ = run_suite()
    if not ok:
        print("baseline suite is already failing, fix that first")
        return 2
    print("baseline: green\n")

    survivors = []
    for label, rel, old, new, guard in MUTATIONS:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if old not in original:
            print(f"SKIP  {label}\n      anchor not found in {rel}, mutation is stale")
            survivors.append(label)
            continue
        try:
            path.write_text(original.replace(old, new, 1), encoding="utf-8")
            passed, output = run_suite()
            if passed:
                print(f"SURVIVED  {label}\n          nothing failed, {guard} is not doing its job")
                survivors.append(label)
            else:
                caught_by_guard = guard in output
                mark = "KILLED" if caught_by_guard else "KILLED*"
                note = "" if caught_by_guard else f"  (failed, but not via {guard})"
                print(f"{mark}    {label}{note}")
        finally:
            path.write_text(original, encoding="utf-8")

    print()
    ok, _ = run_suite()
    print(f"restored baseline: {'green' if ok else 'RED - files not restored cleanly'}")
    if survivors:
        print(f"\n{len(survivors)} mutation(s) survived:")
        for s in survivors:
            print(f"  - {s}")
        return 1
    print(f"\nall {len(MUTATIONS)} mutations killed")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
