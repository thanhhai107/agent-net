# NIKA Procedural Memory README

This document is the canonical design and operations note for the NIKA
cross-session procedural memory module. It replaces the older split notes:

- `docs/memory_inputs.md`
- `docs/memory_module.md`
- `docs/memory_system_research.md`

The goal is not to copy any one paper. The current design combines the useful
parts of A-Mem, LightMem, and MemInsight with a NIKA-specific benchmark safety
boundary: the agent may learn from public context, redacted trajectories,
solver submissions, and numeric evaluation feedback, but must not learn oracle
answers from the benchmark.

## Short Answer

The current implementation is a solid benchmark-safe baseline:

```text
MemInsight-style attributes
+ A-Mem-style atomic procedural notes and typed links
+ LightMem-style post-episode update
+ NIKA numeric score gate
+ PostgreSQL canonical store
+ optional Qdrant semantic index
```

It is not yet a full state-of-the-art memory system. The largest remaining
gaps are learned/semantic topic summarization, sleep-time batch consolidation,
soft attribute matching, calibrated retrieval/ranking, and broader
memory-specific benchmark evaluation.

## System Boundary

NIKA is the benchmark and evaluator. It is not the memory architecture.

```text
benchmark row
  -> start network scenario
  -> inject hidden fault
  -> generate public task
  -> solver investigates the lab
  -> solver submits final answer
  -> evaluator compares submission with ground truth
  -> numeric metrics
```

Memory belongs to the evaluated agent side. The benchmark may provide safe
episode artifacts, but the memory module must not directly read hidden answers
or fault injection internals as learning material.

Allowed learning signals:

```text
public episode context
+ redacted solver trajectory
+ solver submission
+ numeric evaluation feedback
```

Forbidden learning signals:

```text
ground_truth.json
benchmark CSV problem id
problem_names answer labels
failure injection parameters
oracle root-cause labels
correct faulty devices
verify/oracle results
scenario-to-answer mappings
```

If memory sees oracle fields, it stops being an experience-based diagnostic
module and becomes a benchmark answer cache.

## Source Lineage

Papers and repositories reviewed for this design:

| System | Paper | Repository snapshot reviewed |
|---|---|---|
| A-Mem | https://arxiv.org/abs/2502.12110 | `agiresearch/a-mem` at `ceffb860f0712bbae97b184d440df62bc910ca8d` |
| LightMem | https://arxiv.org/abs/2510.18866 | `zjunlp/LightMem` at `26940821cea7265f6a8a5df8b5cdf60e8d2a80ff` |
| MemInsight | https://arxiv.org/abs/2503.21760 | `amazon-science/MemInsight` at `12c6bcbad526b1826b2fdf787dc126584b99c5d5` |

The implementation was compared against source code, not just paper diagrams.
That matters because the code often narrows or simplifies the paper design.

## Architecture Overview

At runtime, memory is a composable wrapper around an existing troubleshooting
agent, not a separate workflow:

```text
public task context
  -> MemoryAugmentedAgent
  -> ProceduralMemoryModule.retrieve()
  -> inject concise memory guidance
  -> selected solver workflow runs
  -> solver writes submission
  -> NIKA evaluator writes numeric metrics
  -> evolve_session_memory()
  -> ProceduralMemoryModule.extract/validate/consolidate()
  -> canonical memory bank + rebuildable indexes
```

The important separation is:

| Phase | Online? | Reads memory? | Writes memory? | Sees oracle? |
|---|---:|---:|---:|---:|
| Retrieval before diagnosis | yes | yes | no | no |
| Agent diagnosis | yes | receives guidance | no | no |
| Evaluation | after submission | no | no | yes, evaluator only |
| Memory evolution | after evaluation | yes | yes | no ground-truth text |
| Sleep-time consolidation target | offline | yes | yes | no |

## Paper/Code Comparison Matrix

| Axis | A-Mem | LightMem | MemInsight | NIKA memory today |
|---|---|---|---|---|
| Main idea | Atomic note graph/evolution | Efficient lifecycle and sleep-time update | Attribute augmentation/retrieval | Benchmark-safe procedural memory |
| Representation | Note with content, metadata, embedding, links | MemoryEntry/topic records | Attribute-value annotated memories | Atomic procedural note with evidence, avoid rules, attributes |
| Extraction | LLM creates note metadata | Compress/segment, then extract | LLM mines attributes | LLM extracts at most six atomic notes from compacted trace |
| Consolidation | LLM updates linked notes | Offline update queues | Mostly absent | Add/corroborate, typed links, limited supersede |
| Retrieval | Vector plus linked neighbors | Qdrant vector search | Attribute or FAISS retrieval | FTS + optional Qdrant + attributes + confidence + graph |
| Ranking | Mostly vector order | Mostly vector order | Attribute match/similarity | Weighted relevance, attributes, confidence, graph, diversity |
| Storage | In-memory dict plus Chroma in code snapshot | Qdrant payload path | JSON/FAISS experiment files | PostgreSQL/SQLite canonical store, Qdrant rebuildable |
| Scheduling | Add-time evolution | Online insert, offline update | Batch scripts | Retrieval online, update post-eval, sleep-time still roadmap |
| Validation | LLM judgment | LLM update decisions | Task evaluation scripts | Numeric NIKA score gate |
| Main risk | Unvalidated mutation | Compression/update loss | Generic attributes | Needs fuller compaction/eval/calibration |

## What We Inherit

### A-Mem

A-Mem's useful idea is the atomic note graph:

```text
one memory = one reusable note
+ generated metadata
+ embedding text
+ links to related notes
+ evolution when new evidence arrives
```

What NIKA adopts:

- atomic procedural notes instead of raw episode dumps;
- graph links between notes;
- relation-aware consolidation;
- memory evolution through refine/support/contradict/same-pattern relations.

What NIKA does differently:

- A-Mem's default code keeps canonical memory largely in process memory with a
  Chroma retriever; NIKA uses PostgreSQL/SQLite as canonical state and treats
  Qdrant as rebuildable.
- A-Mem code uses mostly untyped note links; NIKA has typed links:
  `supports`, `refines`, `contradicts`, `same_pattern`.
- A-Mem mutates neighboring memories via LLM judgment without benchmark
  validation; NIKA gates memory confidence with numeric evaluation metrics.

### LightMem

LightMem's useful idea is the lifecycle:

```text
raw messages
  -> compression / noise filtering
  -> topic segmentation
  -> short-term buffer
  -> extracted long-term memories
  -> online soft insert
  -> offline sleep-time consolidation
```

What NIKA adopts:

- no memory writes before the solver submits an answer;
- memory update after evaluation;
- trajectory compaction before extraction;
- separation between online retrieval and post-episode update;
- the idea that heavy consolidation should move toward sleep-time jobs.

What NIKA does differently today:

- current `compact_trace()` is a deterministic event filter, redactor, and
  topic grouper, not a full learned LightMem segmenter/summarizer;
- current consolidation runs immediately after evaluation, not as a queued
  offline sleep-time batch;
- NIKA adds numeric correctness gates because LightMem is not benchmark-oracle
  aware by itself.

### MemInsight

MemInsight's useful idea is attribute-aware memory retrieval:

```text
memory text alone is not enough;
retrieve using context attributes such as scenario, domain, topic, user state,
protocol, task stage, and evidence type
```

What NIKA adopts:

- every memory carries non-oracle attributes;
- every query is built from public task/scenario context;
- retrieval ranking combines lexical/semantic relevance with attribute match;
- memory is not selected by cosine similarity alone.

What NIKA does differently:

- MemInsight implementation is largely batch augmentation plus JSON/FAISS-style
  retrieval; NIKA stores attributes in canonical memory records and ranking.
- NIKA attributes are network-diagnosis specific: protocols, services,
  topology class, task stage, symptoms, and tools.
- NIKA must prevent attributes from encoding hidden problem labels.

## Current Implementation Map

Core files:

| File | Role |
|---|---|
| `src/agent/memory/models.py` | Data models: attributes, candidates, stored memories, queries, relation decisions |
| `src/agent/memory/attributes.py` | Deterministic non-oracle attribute mining for protocols, services, symptoms, stages, and tools |
| `src/agent/memory/safety.py` | Oracle-leakage guard for extractor/retriever payloads |
| `src/agent/memory/service.py` | Extraction, validation, consolidation, retrieval, context formatting |
| `src/agent/memory/store.py` | Canonical SQLite/PostgreSQL stores, FTS, links, episodes, retrieval logs |
| `src/agent/memory/vector_index.py` | Optional Qdrant semantic index |
| `src/agent/memory/adapter.py` | Composable wrapper around `react`, `plan-execute`, `reflexion` |
| `src/agent/memory/workflow.py` | Post-evaluation memory evolution hook |
| `src/nika/codex_cli/commands/memory.py` | `nika memory run/inspect/health/snapshot/clear` commands |
| `tests/test_hybrid_memory.py` | Unit/integration coverage for current memory behavior |

The module is explicitly organized around:

```text
MemInsight-style context attributes
+ LightMem-style post-episode validation/consolidation
+ A-Mem-style atomic notes and graph links
```

## Memory Representation

Each extracted candidate is an atomic procedural note:

```python
MemoryCandidate = {
    "content": "one reusable diagnostic lesson",
    "applicability": ["when this lesson applies"],
    "evidence_required": ["what to verify before trusting it"],
    "avoid": ["pitfalls or invalid shortcuts"],
    "attributes": {
        "scenarios": [],
        "topology_classes": [],
        "protocols": [],
        "services": [],
        "task_stages": [],
        "symptoms": [],
        "tools": []
    }
}
```

Stored records add lifecycle and provenance fields:

```python
StoredMemory = {
    "memory_id": "...",
    "bank_id": "...",
    "status": "staged | validated | superseded | rejected",
    "confidence": 0.0,
    "source_session_id": "...",
    "version": 1,
    "validation_count": 0,
    "failure_count": 0,
    "created_at": "...",
    "superseded_at": None,
    "superseded_by": None
}
```

There is no public taxonomy like `observation`, `learning`, or `instruction`.
The memory type is always an atomic procedural note; the difference is in
content, evidence requirements, status, confidence, and graph support.

Good memory:

```text
When diagnosing BGP reachability failures, compare session state and route
advertisement at both endpoints before localizing the affected role.
```

Bad memory:

```text
In session 20260626-abc123, pc2 was faulty.
```

The first teaches a reusable diagnostic policy. The second memorizes a
benchmark instance.

## Storage And Indexing

Canonical state:

- PostgreSQL by default through `MEMORY_DATABASE_URL`;
- SQLite only when forced, mainly for tests and local compatibility;
- tables for episodes, memories, typed links, retrieval events, and FTS;
- JSONL snapshots for audit/export/reproducibility.

Secondary indexes:

- PostgreSQL/SQLite full-text search is the lexical fallback;
- Qdrant is optional semantic retrieval;
- Qdrant query text is separately redacted and capped to fit the current
  `result-embed-dr` 512-token embedding limit; lexical search still receives
  the full task plus attribute terms;
- Qdrant failures do not invalidate memory because Qdrant is not the source of
  truth.

Default local services:

```shell
docker compose up -d postgres qdrant
```

Relevant environment variables:

```dotenv
MEMORY_DATABASE_URL=postgresql://nika:nika@localhost:5432/nika_memory

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=nika_memory

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=result-embed-dr
EMBEDDING_DIMENSION=1024
EMBEDDING_API_KEY=${NETMIND_API_KEY}
EMBEDDING_BASE_URL=${NETMIND_BASE_URL}
```

If `EMBEDDING_MODEL` or Qdrant config is missing, retrieval falls back to the
canonical store and FTS.

## Lifecycle

### 1. Before Episode: Retrieval Only

The memory wrapper builds a public query from:

- task description;
- scenario name;
- topology class;
- inferred protocols;
- available diagnostic tool names;
- top-k and token budget.

The wrapper then retrieves validated memories and appends a concise context
block to the agent task:

```text
Prior procedural memories (guidance only; verify every item with current tools):
1. [confidence=0.90] ...
   Applies when: ...
   Verify with: ...
   Avoid: ...
```

This is guidance, not ground truth. The agent must verify every retrieved item
with current tools.

### 2. During Episode: No Writes

During diagnosis the selected agent runs normally:

- `react`
- `plan-execute`
- `reflexion`

The memory module does not update, consolidate, or validate while the solver is
still investigating. This prevents the agent from learning from the current
episode before submitting.

Note: `reflexion` also has attempt-local episodic memory inside one run. That is
separate from this cross-session procedural memory bank.

### 3. After Episode: Numeric-Gated Update

After `eval_metrics.json` exists, `evolve_session_memory()` runs when
`memory_mode=evolve`.

Flow:

```text
messages.jsonl
  -> compact_trace()
  -> LLM extracts atomic candidates
  -> EvaluationEvidence gates candidates
  -> add_or_corroborate()
  -> optional Qdrant upsert
  -> typed relation classification
  -> optional supersede
  -> memory_snapshot.jsonl
  -> memory_update.json
```

The extractor receives:

- redacted task description;
- scenario family/name as context;
- topology class;
- compacted diagnosis trajectory.

The extractor does not receive:

- `problem_names`;
- `ground_truth`;
- raw benchmark answers;
- current episode score text as a reasoning clue.

### 4. Later Episodes: Reuse

Later episodes can retrieve from the same bank in `read` or `evolve` mode.
`read` treats the bank as frozen. `evolve` retrieves, runs, evaluates, and then
updates the bank again.

## Reasoning And Reflection Pipeline

Memory is injected as a small procedural context block at the start of the
solver task. It should influence search strategy, evidence collection, and
avoidance of known bad shortcuts. It should not provide final answers.

For `react` and `plan-execute`, the wrapper leaves the workflow graph intact:

```text
task description
  -> append retrieved procedural memories
  -> original agent workflow
  -> tool calls and reasoning
  -> final submission
```

For `reflexion`, there are two memory layers:

| Layer | Scope | Purpose |
|---|---|---|
| Reflexion episodic memory | within one run/attempt loop | remember failed attempts and change next strategy |
| Procedural memory module | across sessions and benchmark episodes | reuse validated diagnostic lessons |

These layers must stay conceptually separate. Reflexion memory can help the
next attempt in the same session, while procedural memory learns only after the
episode has been evaluated.

The retrieved memory text intentionally says "guidance only; verify every item
with current tools" because network diagnosis must be evidence-driven. A memory
may suggest which checks are worth running, but the current episode's tool
observations decide the answer.

## Extraction Strategy

The current extraction prompt asks for at most six atomic notes and enforces:

- generalize device names, addresses, interfaces, topology-specific values;
- do not store benchmark problem identifiers;
- do not store scenario-to-answer mappings;
- treat root causes as hypotheses requiring evidence;
- do not invent observations absent from the trajectory;
- keep tool names when useful for reproducibility;
- assume ground truth and benchmark scores are unavailable.

Current compaction:

- reads `messages.jsonl`;
- keeps diagnosis-agent events only;
- keeps tool starts, tool ends/errors, and selected LLM/codex turn text;
- groups retained events into diagnostic topics such as routing inspection,
  service inspection, connectivity probes, configuration inspection, and tool
  error recovery;
- emits both compact events and topic-level evidence summaries;
- truncates long payloads;
- drops older events until under `max_chars`;
- redacts IPs, MACs, interface names, and device-like identifiers.

Current gap:

```text
compact_trace() is safe and topic-aware, but it is still deterministic. It is
not yet a learned LightMem-style semantic segmenter plus LLM summarizer.
```

Target compaction:

```text
raw messages.jsonl
  -> remove hidden/oracle fields
  -> redact exact identifiers
  -> filter low-value chatter
  -> segment by diagnostic topic
  -> summarize each topic
  -> extract atomic memory candidates
```

Suggested diagnostic topics:

- initial symptom interpretation;
- topology/routing inspection;
- service inspection;
- failed hypothesis;
- tool error/recovery;
- final reasoning and submission.

## Validation Strategy

NIKA's most important addition is numeric score-gated validation.

The validation input is only:

```python
EvaluationEvidence = {
    "detection_score": float,
    "localization_f1": float,
    "rca_f1": float
}
```

Current gate:

| Episode outcome | Memory action |
|---|---|
| detection=1, localization_f1=1, rca_f1=1 | validate all extracted candidates |
| partial or failed | stage only candidates with `evidence_required` or `avoid` |
| uncheckable candidate from failed/partial run | reject by omission |
| repeated same content in successful episodes | increase validation count and confidence |
| later validated refinement | supersede older lower-confidence memory |

Current confidence formulas:

```python
if fully_successful:
    confidence = 0.72 + 0.18 * aggregate_score
else:
    confidence = 0.30 + 0.25 * aggregate_score
```

When an existing note is corroborated, confidence moves by `+0.08` on success
and `-0.08` on failure. A staged note can become validated after a later
successful corroboration.

This intentionally avoids "LLM says this memory sounds right" as the main
validation mechanism.

## Consolidation And Evolution

Current consolidation:

```text
candidate
  -> canonical add_or_corroborate by content hash
  -> optional semantic upsert
  -> find nearby validated/staged notes by FTS
  -> LLM classifies relation
  -> add typed link
  -> supersede older memory if new validated note refines it with enough confidence
```

Supported link types:

- `supports`
- `refines`
- `contradicts`
- `same_pattern`

Current limitations:

- relation candidates come from local FTS top-3, not a broader vector plus
  attribute plus recency batch;
- contradiction links are used as a ranking penalty, but there is not yet a
  dedicated contradiction-resolution workflow;
- deduplication is content-hash based and may collapse same wording across
  different contexts;
- there is no persistent update-job queue yet;
- rejected candidates are not stored as first-class audit records.

Target consolidation:

```text
sleep-time job
  -> gather recent staged/validated memories
  -> retrieve neighbors by vector + attributes + timestamp
  -> batch classify typed relations
  -> corroborate or demote
  -> supersede with tombstones, not destructive deletes
  -> rebuild FTS/Qdrant indexes
  -> write audit snapshot
```

## Retrieval And Ranking

Current retrieval candidate sources:

- FTS over validated memories;
- optional Qdrant vector search over validated memories;
- fallback to high-confidence recent memories when lexical search has no hit.

Current query attributes:

```python
MemoryQuery = {
    "text": "...",
    "scenario": "...",
    "topology_class": "...",
    "protocols": [...],
    "services": [...],
    "symptoms": [...],
    "task_stage": "diagnosis",
    "tools": [...],
    "top_k": 5,
    "candidate_limit": 20,
    "token_budget": 1500
}
```

Current ranking:

```python
relevance = max(lexical_score, semantic_score)

score = (
    0.40 * relevance
  + 0.30 * attribute_score
  + 0.20 * memory.confidence
  + 0.10 * graph_score
)
```

Selection also applies:

- diversity penalty using Jaccard overlap;
- token-budget limit;
- top-k limit.

Current gap:

- attribute matching is exact and flat;
- Qdrant payload filters are used for bank/status/protocol/service/task-stage,
  but not yet for soft matching or graph expansion;
- graph score is mostly link count plus note structure;
- weights are hand-set, not benchmark-calibrated;
- contradiction penalty exists, but still needs benchmark calibration;
- no dedicated retrieval-evaluation harness yet.

Target retrieval:

```text
candidate_generation =
    lexical content/evidence/tool search
  + vector search over embedding_text
  + attribute filters over protocol/service/topology/task_stage
  + graph expansion from high-confidence supports/refines links

ranking_score =
    semantic_similarity
  + lexical_score
  + attribute_match_score
  + graph_support_score
  + confidence_and_validation_score
  + recency_or_domain_fit_score
  - contradiction_penalty
  - redundancy_penalty
  - token_budget_penalty
```

## Benchmark Input Contract

Memory input must be split into two phases.

### Retrieval Input

Retrieval happens before the solver acts. It must contain public-only
information:

```python
MemoryRetrievalInput = {
    "episode_id": "...",
    "query_text": "...",
    "scenario_name": "dc_clos_bgp",
    "topology_class": "s",
    "topology_summary": {...},
    "protocols": ["bgp"],
    "services": [],
    "task_stage": "diagnosis",
    "limits": {
        "candidate_limit": 20,
        "top_k": 5,
        "token_budget": 1500
    }
}
```

It must not contain:

- injected fault;
- benchmark CSV `problem`;
- `problem_names`;
- `ground_truth.json`;
- correct faulty devices;
- correct root cause;
- failure injection params;
- evaluation score for the current episode;
- any post-episode data.

### Update Input

Update happens after the solver has submitted and evaluator metrics exist:

```python
MemoryUpdateInput = {
    "episode_id": "...",
    "public_context": {...},
    "trajectory": {
        "events": [...],
        "redacted": True,
        "excluded_fields": [
            "ground_truth",
            "problem_names",
            "root_cause_name_from_run_meta",
            "injection_params",
            "verify_result"
        ]
    },
    "submission": {
        "is_anomaly": True,
        "faulty_devices": ["<device>"],
        "root_cause_name": ["solver_submitted_label"]
    },
    "feedback": {
        "detection_score": 1.0,
        "localization_f1": 1.0,
        "rca_f1": 1.0
    }
}
```

The current implementation does not pass `submission` into the extractor; it
uses the diagnosis trajectory and numeric metrics. If future versions use
submission, it must be explicitly marked as solver output, not truth.

## Sanitization And Redaction

Forbidden keys for any memory LLM or embedding payload:

```python
FORBIDDEN_KEYS = {
    "problem",
    "problem_names",
    "root_cause_name",
    "ground_truth",
    "ground_truth_path",
    "faulty_devices_from_ground_truth",
    "failure_injections",
    "injection_params",
    "verify_result",
    "requested_overrides",
    "resolved_params",
}
```

Persistent memory should redact or generalize:

| Pattern | Replacement |
|---|---|
| IPv4/IPv6 address | `<network-address>` |
| MAC address | `<mac-address>` |
| interface name | `<interface>` |
| concrete host/router/switch names | `<device>` or a role |
| session id | `<session_id>` |
| container id | `<container_id>` |
| benchmark problem id | never store |

Tool names may be preserved when they make the procedure reproducible.

## Supported Modes

CLI surface:

```shell
# baseline: omit memory flags
--memory <bank-id>
--memory-read <bank-id>
--memory-k 5
--memory-tokens 1500
```

Mode behavior:

| Mode | Retrieval before episode | Update after evaluation | Intended use |
|---|---:|---:|---|
| `off` | no | no | baseline |
| `read` | yes | no | evaluate a frozen memory bank |
| `evolve` | yes | yes | online continual-memory experiment |

Memory-compatible agents:

- `react`
- `plan-execute`
- `reflexion`

When memory is enabled, problem-tool hints are disabled for those workflows so
that memory experiments do not accidentally reuse oracle-flavored hints.

## Operating Commands

Check services:

```shell
docker compose up -d postgres qdrant
nika memory health --bank experiment-01
```

Run a quick memory-only evolution benchmark:

```shell
nika memory run --bank experiment-01 --limit 4
```

Evaluate a frozen bank with a different agent:

```shell
nika memory run --bank experiment-01 --read --limit 4 -a plan-execute
```

Run an agent on the current session with memory:

```shell
nika agent run -a reflexion \
  --memory experiment-01
```

Equivalent advanced benchmark command:

```shell
nika benchmark run --file benchmark/benchmark_test.csv \
  -a react --memory experiment-01
```

Inspect and export:

```shell
nika memory inspect --bank experiment-01
nika memory snapshot --bank experiment-01
nika memory clear --bank experiment-01 -y
```

Validate the memory module:

```shell
uv run --with pytest pytest tests/test_hybrid_memory.py -q
```

## Smoke Benchmark Evidence

The latest local memory-only smoke run used the simplified command:

```shell
nika memory run --bank memory-only-tight --limit 3 --reset-bank -n 50
```

Observed artifacts:

| Artifact | Value |
|---|---|
| Log | `runtime/memory/runs/memory-only-tight.log` |
| Snapshot | `runtime/memory/memory-only-tight.snapshot.jsonl` |
| Sessions | `20260629-003738-b3aa88`, `20260629-003855-8cb7fa`, `20260629-004002-d413d3` |
| Completion | `total=3 completed=3 failed=0` |
| Memory update | 6 candidates accepted per episode, 18 staged memories total |
| Qdrant context warnings | 0 after semantic query cap |
| Oracle leak scan | 0 hits for `ground_truth`, `problem_names`, `root_cause_name`, and current problem ids |

Per-case numeric metrics from this smoke run:

| Session | Scenario | Problem | Detection | Localization F1 | RCA F1 | Notes |
|---|---|---|---:|---:|---:|---|
| `20260629-003738-b3aa88` | `ospf_enterprise_dhcp` | `dns_record_error` | 1.0 | 0.0 | 1.0 | diagnosis found the root-cause class, localization remained imperfect |
| `20260629-003855-8cb7fa` | `dc_clos_bgp` | `host_crash` | -1.0 | -1.0 | -1.0 | no final submission; memory still staged cautious notes only |
| `20260629-004002-d413d3` | `dc_clos_bgp` | `host_ip_conflict` | 0.0 | 0.0 | 0.0 | agent submitted no anomaly; memory remains staged |

Interpretation:

- The memory lifecycle works end to end: retrieve, run, evaluate, extract,
  gate, store, snapshot.
- Failed or partial episodes produce staged memories, not validated memories.
  That is the intended benchmark-safe behavior.
- The run evaluates the memory pipeline, not proof of improved diagnosis
  quality. Quality should be measured by paired memory off/read/evolve
  experiments on larger fixed suites.

## Online Evolution Protocol

`memory_mode=evolve` runs sequentially:

```text
episode 1 retrieve from memory_so_far
episode 1 run solver
episode 1 evaluate
episode 1 update memory

episode 2 retrieve from memory_so_far
episode 2 run solver
episode 2 evaluate
episode 2 update memory
```

Benchmark CSV execution is sequential by design, so a future episode cannot
update memory before an earlier episode retrieves from the bank.

Use a unique bank per experiment condition, model, seed, and benchmark suite.

## Evaluation Plan

Primary benchmark metrics:

- detection score;
- localization F1;
- RCA F1;
- task success rate;
- steps;
- tool calls;
- tool errors;
- token usage;
- wall-clock time.

Memory-specific metrics:

- retrieval count and selected memory ids;
- retrieval precision by public attributes;
- memory acceptance rate;
- staged/validated/superseded counts;
- contradiction/refinement rate;
- duplicate or near-duplicate memory rate;
- harmful retrieval rate;
- oracle-leak rate;
- update latency and extraction token cost.

Recommended ablations:

| Variant | Purpose |
|---|---|
| memory off | baseline |
| read-only frozen bank | measure retrieval-only benefit |
| evolve full | online continual-memory method |
| evolve without attributes | isolate MemInsight-style attributes |
| evolve without graph links | isolate A-Mem graph contribution |
| evolve without semantic index | measure FTS-only fallback |
| evolve without score gate | should be unsafe; useful only as a controlled ablation |
| evolve with topic compaction | isolate LightMem-style compaction |

Required safety tests:

- extractor never receives `problem_names`;
- extractor never receives `ground_truth`;
- retrieval input has no post-episode data;
- long-term memory content redacts concrete identifiers;
- benchmark CSV execution remains sequential;
- memory context tells agent to verify with tools.

Current tests already cover parts of this, including no `problem_names` or
`ground_truth` in extractor kwargs and sequential benchmark execution.

## Current Fit Assessment

The implementation is suitable as:

- a benchmark-safe memory baseline;
- a real cross-session procedural memory module;
- a foundation for A-Mem/LightMem/MemInsight hybrid research;
- an experiment harness for off/read/evolve comparisons.

It is not yet complete enough to call state-of-the-art because:

- compaction is topic-grouped but not yet learned/semantic summarized;
- query attribute extraction is deterministic and useful, but still lacks soft
  taxonomy/synonym handling;
- relation classification is local and immediate;
- sleep-time batch consolidation is not implemented as a job system;
- retrieval weights are not calibrated by benchmark data;
- graph ranking is shallow;
- rejected candidates are not first-class audit records;
- provenance lacks trace-span-level evidence;
- memory evaluation needs broader ablations and leak regression tests.

## Roadmap

### P0: Safety And Reproducibility

- Add a sanitizer object that builds explicit retrieval/update payloads.
- Add `assert_no_oracle_leakage()` tests over payloads, prompts, embedding text,
  snapshots, and retrieval logs.
- Store per-memory provenance hashes and trace-span references.
- Store rejected candidates or at least rejection counts for audit.

### P1: LightMem-Style Compaction

- Replace fixed truncation with topic segmentation.
- Summarize each diagnostic topic separately.
- Preserve tool observations with higher priority than LLM self-talk.
- Add optional compression only when it does not remove network-critical details.

### P2: Attribute Mining And Retrieval

- Extend the safe deterministic attribute miner to topology family and evidence
  type.
- Add soft attribute matching with synonyms and hierarchy.
- Expand Qdrant payload filters beyond protocol/service/task-stage.
- Calibrate contradiction penalties and relation-aware graph expansion.

### P3: Sleep-Time Consolidation

- Add persistent update jobs.
- Batch relation classification across vector and attribute neighbors.
- Promote/demote based on repeated evidence.
- Supersede with versioned tombstones.
- Rebuild indexes from canonical PostgreSQL.

### P4: Benchmark Evaluation

- Run controlled off/read/evolve experiments.
- Track memory-specific metrics in benchmark outputs.
- Add ablations for graph, attributes, semantic index, and compaction.
- Report harmful memory retrieval and oracle-leak rate.

## Design Stance

The target state is:

```text
A-Mem atomic graph
+ LightMem sleep-time lifecycle
+ MemInsight attributes
+ NIKA numeric score gate
+ strict anti-oracle provenance
+ PostgreSQL canonical store
+ Qdrant/FTS rebuildable indexes
+ retrieval/ranking audit logs
```

This keeps the strongest published ideas while adding what the three systems do
not provide together: benchmark-safe validation, explicit provenance, typed
versioned memory, and evaluation-driven confidence.
