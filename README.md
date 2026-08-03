# support-agent

A customer support triage agent built as an explicit LangGraph state graph.

Generating the reply is the easy part. The work in production is everything
around it: deciding a message does not need an answer at all, catching a draft
that quotes a refund figure nobody retrieved, keeping a suppressed fast path
from silently dropping a customer, and stopping at a human before anything is
sent. This repo is those decisions wired as a graph, with a test on each.

```bash
pip install -e ".[dev]"
python demo.py                      # nine scenarios, no API key required
pytest                              # 85 tests
python scripts/mutation_check.py    # breaks the logic on purpose to check the tests notice
```

## The graph

```mermaid
flowchart TD
    START([message arrives]) --> ingest
    ingest --> dedup{dedup}
    dedup -->|already handled| ENDdup([end: duplicate])
    dedup -->|new| classify[classify + lifecycle stage]
    classify --> triage{worth answering?}
    triage -->|no| ENDskip([end: skipped])
    triage -->|yes| canned{pre-approved reply?}
    canned -->|hit| policy
    canned -->|suppressed or no entry| retrieve
    retrieve --> draft[draft]
    draft --> policy{policy check}
    policy -->|fails, budget left| retrieve
    policy -->|budget spent| escalate([end: escalated])
    policy -->|passes| approval[human approval]
    approval -->|approve| send([end: sent])
    approval -->|reject| reject([end: rejected])
```

Three properties are why this is a graph rather than a chain.

Dedup and triage both run before classification, so a repeated message or a
one-word greeting from a contact who has never paid costs nothing. Each records
why it stopped, which makes a skipped message a decision rather than an
accident.

Canned replies can be suppressed for a given contact, and when that happens the
request falls through to the model. Returning early on a blocked fast path
leaves customers unanswered while the logs still look clean.

A failed policy check feeds the specific violations into a fresh draft. How many
attempts you get comes from the contact's lifecycle stage, and once that runs
out the run escalates instead of handing a reviewer something still broken.

## Lifecycle stages

Stage is computed before generation and decides how much effort the message is
worth.

| Priority | Stage | Condition |
|---|---|---|
| 1 | `advocate` | high lifetime value on a live plan |
| 2 | `dormant` | silent past the threshold, currently 72h |
| 3 | `customer` | paid or enterprise plan, not expired |
| 4 | `intent` | buying signal |
| 5 | `active` | engaged past a message-count floor |
| 6 | `new` | default |

Dormancy is checked ahead of buying signals deliberately. Someone reappearing
after nine days is a re-engagement, and treating it as a hot lead reads badly.
Transitions are constrained as well, so one bad signal read cannot drop a paying
customer back into onboarding.

## Policy layer

The rules that decide whether text reaches a customer are regexes with a test
each, rather than a second model call. Reproducibility matters more than nuance
here: the same draft has to get the same verdict every time, and someone
auditing an incident has to be able to read the rule that allowed it.

- commitments the agent is not authorised to make (`guarantee`, `we promise`, `refunded today`)
- quantities that do not appear in the retrieved context
- legal or medical framing
- personal data echoed back into the channel
- length

Grounding compares whole extracted quantities. `$250` passes when `$250` was
retrieved and fails when it was not. A substring test would let `$25` through
against a context that only ever mentioned `$250`.

## Untrusted input

The prompt format delimits fields with markers like `[MESSAGE]`. A customer who
writes `[/MESSAGE]` into their message can close a field early and open a fake
one, so their text arrives at the model looking like an instruction from you.
Untrusted strings go through `neutralise()` before interpolation: the message,
the conversation history, and retrieved passages, since support corpora are
usually built from tickets and community posts and are user-submitted at one
hop of indirection.

Ordinary brackets are left alone, so `[Errno 2]` in a bug report still reads
correctly.

## Human in the loop

The approval node calls LangGraph's `interrupt`, which checkpoints the graph.
The process can die at that point and a different worker can resume it hours
later with the reviewer's decision. Approve, approve with an edit, or reject;
anything the router does not recognise as an approval fails closed.

Every node appends to a `trace` list that accumulates rather than overwrites, so
a finished run can be read back without a tracing backend attached:

```
  . ingest: 50 chars
  . dedup: miss
  . classify: intent=billing stage=customer (plan=paid, active)
  . triage: proceed
  . canned: no entry, falling through to model
  . retrieve: 2 passages
  . draft: 58 chars
  . policy: 3 issue(s) - makes a commitment the agent cannot authorise; states
            '$250', which is not in the retrieved context; states '3 business
            days', which is not in the retrieved context
  . retrieve: 2 passages
  . redraft #1: 78 chars
  . policy: pass
  . approval: approve
  . send: dispatched
```

## Testing

The default provider is a deterministic stub, so every branch is reachable with
no network and no API key. `ScriptedStub` plays back fixed completions for paths
a rule-based stub will never reach on its own: a draft that breaks policy, a
malformed classification, a model that repeats the same mistake until the budget
runs out.

A passing suite says little until you have watched it fail for the right reason,
so `scripts/mutation_check.py` removes one safety property at a time and checks
that the test named as its guard is the one that breaks:

```
KILLED    fallback edge removed: suppressed canned reply dead-ends
KILLED    approval fails open: unknown decision is treated as approval
KILLED    rewrite loop unbounded: budget check removed
KILLED    dormancy no longer outranks buying signals
KILLED    grounding back to a substring test: '$25' passes against '$250'
KILLED    prompt field markers no longer stripped from untrusted text

all 6 mutations killed
```

The last two exist because those were real bugs, found in a review before this
was published and written up in [`docs/security-review.md`](docs/security-review.md).
Both were sitting in the features this README talks about most.

## Wiring in real components

Three seams, one method each:

```python
Deps(
    provider=AnthropicProvider(),   # or any object with .complete()
    knowledge=KnowledgeBase(),      # .search(intent, query) -> list[str]
    dedup=DedupStore(),             # .seen(key) / .remember(key)
)
```

`build_graph(deps, checkpointer=...)` accepts any LangGraph checkpointer.
In-memory for tests, Postgres in production, nothing else changes.

```bash
pip install -e ".[anthropic,dev]"
export ANTHROPIC_API_KEY=...
export SUPPORT_AGENT_PROVIDER=anthropic
```

## Scope

This is a control-flow reference, not a finished product. Retrieval is a dict
lookup, the canned library lives in memory, and `send` only marks the outcome.
Those are the three places to replace first. What is worth copying is the graph
shape, the policy layer, and the way the tests are held to account.

Python 3.11 or newer. Built against langgraph 1.2.

MIT licensed.
