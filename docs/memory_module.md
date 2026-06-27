# Thiết kế Memory Module

Trong thiết kế này, chỉ tập trung vào ba paper:

| Paper | Vai trò |
|---|---|
| **MemInsight** | Attribute-aware retrieval: retrieve đúng memory theo scenario/protocol/topology/task |
| **LightMem** | Offline consolidation: compact trajectory, filter noise, update sau episode |
| **A-Mem** | Atomic memory graph: lưu bài học nhỏ, có liên kết và evolution |

## 1. Nguyên tắc cốt lõi

NIKA không phải là agent architecture. NIKA chỉ nên được xem là:

```text
benchmark sinh incident episode
→ cho solver/agent tương tác với network lab
→ nhận final submission
→ dùng ground truth để chấm điểm
→ xuất numeric metrics
```

Memory module chỉ được học từ:

```text
public episode context
+ redacted solver trajectory
+ solver submission
+ numeric evaluation feedback
```

Memory module **không được học trực tiếp** từ:

```text
ground_truth.json
problem / problem_names
failure injection params
oracle root-cause labels
correct faulty devices
verify result
mapping scenario → answer
```

Nếu memory nhìn thấy các oracle fields này, nó không còn là “agent học từ kinh nghiệm”, mà biến thành cache đáp án benchmark.

---

## 2. Tổng quan pipeline

```text
Episode t
  ↓
[Before episode]
Build public retrieval input
  ↓
MemInsight-style attribute-aware retrieval
  ↓
Inject selected memories into agent prompt
  ↓
Agent diagnoses NIKA incident
  ↓
Agent submits final answer
  ↓
NIKA evaluator computes numeric scores
  ↓
[After episode]
LightMem-style trajectory compaction
  ↓
A-Mem-style atomic memory extraction
  ↓
NIKA score-gated validation
  ↓
A-Mem-style link generation / consolidation
  ↓
Updated memory bank

Episode t+1
  ↓
Retrieve improved memory
```

Self-evolution xảy ra ở external memory bank:

```text
memory mới được thêm
memory cũ được link/refine/supersede
confidence/status thay đổi
retrieval index thay đổi
validation_count và graph support tăng khi bài học được corroborate
```

## 2.1. Storage memory

Thiết kế runtime hiện tại dùng:

```text
PostgreSQL self-hosted bằng Docker
  → canonical store cho memory records, provenance, status, version, graph links

Qdrant self-hosted bằng Docker
  → rebuildable semantic vector index cho candidate retrieval

JSONL snapshot
  → audit/export/reproducibility artifact
```

Qdrant cũng không phải nguồn sự thật. Nếu Qdrant chưa sẵn sàng, retrieval vẫn
có thể chạy bằng PostgreSQL full-text/lexical search; semantic index có thể
rebuild lại từ PostgreSQL bất kỳ lúc nào.

Kiểm tra runtime:

```shell
docker compose up -d postgres qdrant
nika memory health --bank experiment-01
```

---

## 3. Vai trò của từng paper

## 3.1. MemInsight dùng để retrieve đúng memory

### Ý tưởng lấy từ MemInsight

MemInsight đề xuất augment memory bằng các **semantic/contextual attributes**, sau đó dùng attribute đó để hỗ trợ retrieval.

Trong NIKA, retrieval không nên chỉ dựa vào cosine similarity của text. Hai incident có thể cùng có “packet loss” nhưng root cause khác nhau:

```text
packet loss + BGP session fail       → routing/BGP issue
packet loss + queue buildup          → congestion/microburst
packet loss + HTTP connection surge  → DoS/service overload
packet loss + interface error        → link/interface issue
```

Vì vậy cần retrieve theo attribute.

### Query attributes từ NIKA

Trước mỗi episode, từ public context, tạo retrieval attributes:

```python
query_attributes = {
    "scenario_name": "ospf_enterprise_dhcp",
    "topology_class": "s",
    "topology_family": "enterprise",
    "protocols": ["ospf", "dhcp"],
    "services": ["dhcp", "dns"],
    "task_stage": "diagnosis",
    "goal": "localization_and_rca"
}
```

### Memory attributes

Mỗi memory cũng phải có attributes:

```python
memory_attributes = {
    "scenario_family": "enterprise",
    "topology_class": "small",
    "protocols": ["ospf", "dhcp"],
    "services": ["dhcp"],
    "symptoms": ["client_no_lease", "service_unavailable"],
    "tools": ["get_host_net_config", "systemctl_ops", "netstat"],
    "evidence_type": ["lease_state", "service_status"],
    "task_stage": "rca"
}
```

### Retrieval score đề xuất

```text
retrieval_score =
    semantic_similarity
  + protocol_match
  + topology_match
  + service_match
  + symptom_match
  + tool_or_evidence_match
  + memory_confidence
  - redundancy_penalty
```

Ví dụ:

```text
Current episode:
  OSPF enterprise + DHCP + diagnosis

Nên retrieve:
  "When DHCP clients fail in an OSPF enterprise topology,
   check client lease state and DHCP server process before assuming routing failure."

Không nên retrieve:
  "When BGP route advertisement fails in Clos topology..."
```

### Pseudocode

```python
def retrieve_memories(retrieval_input, memory_bank, top_k=5):
    query_attrs = extract_query_attributes(retrieval_input)

    candidates = filter_by_attributes(
        memory_bank,
        protocols=query_attrs["protocols"],
        services=query_attrs["services"],
        topology_family=query_attrs["topology_family"],
        task_stage=query_attrs["task_stage"],
    )

    scored = []
    for memory in candidates:
        score = (
            semantic_similarity(retrieval_input["task_description"], memory["content"])
            + match_score(query_attrs, memory["attributes"])
            + memory["confidence"]
            - redundancy_penalty(memory, scored)
        )
        scored.append((score, memory))

    return select_top_k(scored, top_k)
```

---

## 3.2. LightMem dùng để compact trajectory và update offline

### Ý tưởng lấy từ LightMem

LightMem nhấn mạnh ba điểm:

1. Raw interaction chứa nhiều noise.
2. Cần group thông tin theo topic thay vì dùng fixed window cứng.
3. Memory update nên tách khỏi online inference bằng sleep-time/offline update.

Trong NIKA, trajectory trong `messages.jsonl` có thể rất dài:

```text
LLM reasoning
tool calls
tool outputs
tool errors
intermediate observations
submission formatting
final answer
```

Không nên đưa nguyên trajectory vào memory generator.

### Pipeline compact trajectory

```text
raw messages.jsonl
  ↓
remove hidden/oracle fields
  ↓
redact hostname/IP/interface/MAC
  ↓
filter noise
  ↓
segment by diagnostic topic
  ↓
summarize each topic
  ↓
feed compacted trace into A-Mem extractor
```

### Topic segmentation theo pha chẩn đoán

Ví dụ:

```python
compacted_trace = [
    {
        "topic": "initial_symptom_interpretation",
        "summary": "Agent identified DHCP-related service failure symptoms."
    },
    {
        "topic": "routing_inspection",
        "summary": "OSPF adjacency and route table appeared healthy."
    },
    {
        "topic": "service_inspection",
        "summary": "DHCP server process was abnormal and clients had no valid leases."
    },
    {
        "topic": "final_rca",
        "summary": "Agent submitted DHCP service failure as RCA."
    }
]
```

### Online/offline separation

Trong episode:

```text
agent chỉ retrieve memory
agent không update memory
agent không consolidate memory
```

Sau episode:

```text
compact trajectory
extract memory candidates
validate bằng numeric scores
deduplicate
link/refine/contradict
persist memory
```

Điều này giữ benchmark sạch. Agent không được học từ chính episode hiện tại trước khi submit.

### Pseudocode

```python
def compact_trajectory(messages_jsonl):
    safe_events = []

    for event in messages_jsonl:
        if contains_oracle_field(event):
            continue

        if is_submission_format_only(event):
            continue

        redacted = redact_identifiers(event)
        important = filter_low_value_content(redacted)

        if important:
            safe_events.append(important)

    topic_groups = segment_by_diagnostic_topic(safe_events)
    summaries = [summarize_topic(group) for group in topic_groups]

    return summaries
```

---

## 3.3. A-Mem dùng để lưu atomic memory và evolve memory graph

### Ý tưởng lấy từ A-Mem

A-Mem tổ chức memory thành các **atomic notes** có:

```text
content
timestamp
keywords
tags
contextual description
embedding
links
```

A-Mem cũng có:

```text
link generation
memory evolution
memory graph
```

Trong NIKA, mỗi memory nên là một bài học nhỏ, độc lập, reusable.

### Memory tốt

```text
When diagnosing DHCP failures in an OSPF enterprise topology,
check DHCP lease state and DHCP server process before assuming routing failure.
```

### Memory xấu

```text
In session 20260626-abc123, pc2 was faulty.
```

Memory xấu học thuộc instance, không học troubleshooting skill.

---

## 4. Memory schema đề xuất

```python
MemoryRecord = {
    "memory_id": "...",

    "content": "...",

    "applicability": [
        "when DHCP clients fail in an OSPF enterprise topology"
    ],

    "evidence_required": [
        "verify client lease state",
        "verify DHCP server process status"
    ],

    "avoid": [
        "do not conclude X without checking Y"
    ],

    "attributes": {
        "scenarios": [...],
        "topology_classes": [...],
        "protocols": [...],
        "services": [...],
        "symptoms": [...],
        "task_stages": [...],
        "tools": [...]
    },

    "confidence": 0.0,

    "status": "staged | validated | superseded | rejected",
    "validation_count": 0,
    "failure_count": 0,

    "provenance": {
        "episode_id": "...",
        "scores": {
            "detection_score": 1.0,
            "localization_f1": 1.0,
            "rca_f1": 1.0
        }
    },

    "links": [
        {"type": "supports", "target": "..."},
        {"type": "refines", "target": "..."},
        {"type": "contradicts", "target": "..."},
        {"type": "same_pattern", "target": "..."}
    ]
}
```

---

## 5. Atomic note patterns nên có

Memory không dùng taxonomy theo loại bài học. Mỗi memory là một **atomic procedural note** độc lập. Khác biệt nằm ở nội dung note, evidence cần kiểm chứng, `avoid`, confidence/status và graph support.

| Pattern | Sinh từ đâu | Ví dụ |
|---|---|---|
| **Evidence-pattern note** | Tool observations | `DHCP failure can appear while OSPF adjacency is healthy.` |
| **Caution/avoid note** | Failed/partial episode | `Do not conclude DNS error from HTTP failure alone.` |
| **Validated procedural note** | Full successful episode | `For BGP reachability failure, check BGP session and route advertisement first.` |
| **Corroborated graph-supported note** | Nhiều episode/link cùng support | `When diagnosing DHCP, check lease, server process, resolver config before RCA.` |

---

## 6. Score-gated validation bằng NIKA evaluator

Đây là đóng góp riêng quan trọng nhất.

Memory không được validate bằng cảm giác của LLM. Memory phải được validate bằng numeric feedback từ NIKA:

```python
feedback = {
    "detection_score": 1.0,
    "localization_f1": 1.0,
    "rca_f1": 1.0,
    "steps": 12,
    "tool_calls": 8,
    "tool_errors": 0
}
```

### Gate đề xuất

| Kết quả episode | Memory action |
|---|---|
| `detection=1`, `localization_f1=1`, `rca_f1=1` | validate atomic procedural notes |
| partial success | stage cautious/checkable notes only |
| full failure | stage avoid-rule only, hoặc reject uncheckable claims |
| repeated validated success | tăng confidence và validation_count |
| contradiction với memory cũ | add link `contradicts` hoặc `refines` |

### Pseudocode

```python
def validate_memory(candidate, feedback):
    det = feedback["detection_score"]
    loc = feedback["localization_f1"]
    rca = feedback["rca_f1"]

    if det == 1.0 and loc == 1.0 and rca == 1.0:
        candidate["status"] = "validated"
        candidate["confidence"] += 0.30

    elif det == 1.0 and (loc > 0.0 or rca > 0.0):
        candidate["status"] = "staged"
        candidate["confidence"] += 0.10

    else:
        candidate["status"] = "staged"
        candidate["confidence"] += 0.05

    return candidate
```

### Điều không được làm

Nếu episode fail:

```python
submission = {"root_cause_name": ["dns_record_error"]}
rca_f1 = 0.0
```

Memory không được suy ra:

```text
The correct root cause is X.
```

Memory chỉ được học:

```text
Do not conclude DNS record error from HTTP failure alone; verify resolver output first.
```

---

## 7. Link generation và consolidation theo A-Mem

Khi có memory mới, tìm memory gần nhất và tạo link:

```python
links = [
    "supports",
    "refines",
    "contradicts",
    "same_pattern"
]
```

### Quy tắc an toàn

Không overwrite memory cũ trực tiếp.

```text
Memory mới supports memory cũ
Memory mới refines memory cũ
Memory mới contradicts memory cũ
Memory mới có thể supersede memory cũ nếu relation là refines và confidence đủ cao
```

### Ví dụ

Memory A:

```text
Ping failure often indicates a link issue.
```

Memory B:

```text
In BGP scenarios, ping failure with missing routes should first trigger BGP route inspection.
```

Link:

```text
B refines A
```

### Pseudocode

```python
def consolidate_memory(candidate, memory_bank):
    neighbors = retrieve_nearby_memories(candidate, memory_bank)

    for old in neighbors:
        relation = classify_relation(candidate, old)

        if relation in ["supports", "refines", "contradicts", "same_pattern"]:
            add_link(candidate, old, relation)

        if relation == "same_pattern":
            old["confidence"] += 0.05

        if relation == "refines" and candidate["confidence"] > old["confidence"]:
            add_link(candidate, old, "refines")

        if relation == "refines" and candidate["confidence"] >= old["confidence"]:
            old["status"] = "superseded"

    memory_bank.add(candidate)
    return memory_bank
```

---

## 8. Input contract từ NIKA sang memory module

## 8.1. Retrieval input trước episode

Chỉ chứa public information:

```python
MemoryRetrievalInput = {
    "episode_id": "...",
    "scenario_name": "ospf_enterprise_dhcp",
    "topology_class": "s",
    "scenario_params_public": {...},
    "topology_summary": {
        "family": "enterprise",
        "routing": "ospf",
        "host_config": "dhcp",
        "scale": "small"
    },
    "task_description": "...",
    "protocols": ["ospf", "dhcp"],
    "services": ["dhcp", "dns"],
    "task_stage": "diagnosis"
}
```

Không được chứa:

```text
problem
problem_names
ground_truth
correct faulty devices
correct root cause
failure injection params
evaluation score của episode hiện tại
post-episode data
```

## 8.2. Update input sau episode

```python
MemoryUpdateInput = {
    "episode_id": "...",
    "public_context": {
        "scenario_name": "ospf_enterprise_dhcp",
        "topology_class": "s",
        "topology_summary": {...},
        "task_description": "...",
        "protocols": ["ospf", "dhcp"],
        "services": ["dhcp", "dns"]
    },
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
        "faulty_devices": ["<device_A>"],
        "root_cause_name": ["dns_record_error"]
    },
    "feedback": {
        "detection_score": 1.0,
        "localization_f1": 1.0,
        "rca_f1": 1.0,
        "steps": 12,
        "tool_calls": 8,
        "tool_errors": 0
    }
}
```

---

## 9. Sanitizer bắt buộc

Memory module không nên tự đọc bừa `run.json`, `ground_truth.json`, benchmark CSV, hoặc session store.

Phải có adapter/sanitizer:

```python
class BenchmarkMemoryAdapter:
    def build_retrieval_input(run_meta: dict) -> MemoryRetrievalInput:
        ...

    def build_update_input(
        run_meta: dict,
        messages_jsonl: list[dict],
        submission: dict | None,
        eval_metrics: dict,
    ) -> MemoryUpdateInput:
        ...

    def assert_no_oracle_leakage(payload: dict) -> None:
        ...

    def snapshot_episode_input(payload: dict, output_path: str) -> None:
        ...
```

Forbidden keys:

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

---

## 10. Redaction rules

Persistent memory nên tránh exact identifiers.

| Pattern | Replace with |
|---|---|
| IPv4/IPv6 | `<ip>` |
| MAC address | `<mac>` |
| `eth0`, `ens3`, `r1-eth0` | `<interface>` |
| concrete host/router/switch names | `<device_A>`, `<router_A>`, `<host_A>` |
| session id | `<session_id>` |
| container id | `<container_id>` |
| benchmark problem id | `<problem_id>` |

Trong episode-local trace có thể giữ raw identifiers để debug, nhưng khi lưu long-term memory phải abstract hóa.

---

## 11. Evaluation metrics

Tài liệu này không định nghĩa biến thể so sánh bắt buộc. Memory module chính là full method:

```text
MemInsight-style retrieval
+ LightMem-style offline consolidation
+ A-Mem-style atomic memory graph
+ NIKA score-gated validation
```

Metrics:

```text
detection accuracy
localization F1
RCA F1
steps
tool calls
tool errors
token usage
time to resolution
memory retrieval precision
memory update acceptance rate
harmful memory retrieval rate
```

Kỳ vọng cải thiện mạnh nhất ở:

```text
localization F1
RCA F1
tool-call efficiency
```

Detection có thể không tăng nhiều vì detection thường dễ hơn localization/RCA.

---

## 12. Online evolution protocol

Nếu chạy `memory_mode=evolve`, phải chạy sequential:

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

Không nên chạy parallel khi memory bank đang evolve.

Quy tắc:

```text
memory_mode=evolve → sequential
memory_mode=read   → có thể parallel nếu memory bank frozen
mỗi experiment/seed → memory bank riêng
```

---

## 13. Folder structure đề xuất

```text
nika_memory/
  adapter/
    nika_adapter.py
    sanitizer.py
    redactor.py

  retrieval/
    attribute_extractor.py      # MemInsight-style
    retriever.py
    reranker.py

  trajectory/
    compactor.py                # LightMem-style
    segmenter.py
    summarizer.py

  memory/
    schema.py                   # A-Mem-style
    extractor.py
    validator.py
    graph.py
    consolidator.py
    store.py

  evaluation/
    leakage_check.py
    memory_metrics.py

  configs/
    memory.yaml

  scripts/
    run_evolve.py
    run_read.py
    inspect_memory.py
```