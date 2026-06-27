# Inputs from NIKA Benchmark to a Memory Module

## Mục tiêu

Tài liệu này định nghĩa một cách sạch sẽ: nếu bỏ qua toàn bộ agent implementation, tool-module và memory-module hiện tại, thì bản thân **NIKA benchmark** có thể cung cấp những gì làm input cho một memory module.

Quan điểm chính:

> NIKA không phải là kiến trúc agent. NIKA là benchmark sinh incident episode, cho solver tương tác với network lab, nhận final submission, rồi chấm điểm bằng ground truth.

Vì vậy, input hợp lệ cho memory module không nên là toàn bộ trạng thái nội bộ của NIKA, mà là một **episode record đã được phân quyền rõ ràng**.

Memory module chỉ nên học từ:

- public episode context;
- redacted solver trajectory;
- solver submission;
- numeric evaluation feedback.

Memory module không nên học trực tiếp từ:

- ground truth text;
- injected problem id;
- failure injection parameters;
- oracle-only root cause labels;
- mapping cụ thể giữa benchmark scenario và đáp án.

Nếu không giữ ranh giới này, memory module sẽ không còn là “agent học từ kinh nghiệm”, mà sẽ biến thành cache đáp án benchmark.

---

## 1. NIKA sinh ra một episode như thế nào?

Một episode của NIKA, nhìn ở mức benchmark thuần, có chuỗi sau:

```text
benchmark row
  ↓
start network scenario
  ↓
inject hidden fault
  ↓
generate public task description
  ↓
solver investigates the lab
  ↓
solver submits answer
  ↓
evaluator compares submission with ground truth
  ↓
numeric metrics
```

Trong repo hiện tại, các artifact tương ứng thường nằm dưới:

```text
results/<session_id>/
  run.json
  messages.jsonl
  submission.json
  ground_truth.json
  eval_metrics.json
  llm_judge.json              # optional
  memory_update.json          # nếu memory evolution bật
  memory_snapshot.jsonl       # nếu memory evolution bật
```

Với một memory module độc lập, các file quan trọng nhất là:

| Artifact | Nội dung chính | Vai trò với memory |
|---|---|---|
| `run.json` | episode metadata, scenario, topology size, task description, timing, agent/model metadata; cũng có thể chứa oracle fields như `problem_names` | lấy public context có chọn lọc |
| `messages.jsonl` | trajectory: reasoning events, tool calls, tool outputs, errors | nguồn chính để extract procedural memory |
| `submission.json` | final solver answer: anomaly, faulty devices, root cause names | dùng sau episode để hiểu solver đã kết luận gì |
| `ground_truth.json` | đáp án thật của benchmark | chỉ evaluator dùng, không đưa cho memory generator |
| `eval_metrics.json` | detection/localization/RCA scores, steps, tool calls, token counts | numeric feedback để validate/stage/reject memory |
| `llm_judge.json` | optional LLM-as-judge feedback | chỉ nên dùng phụ trợ, không làm primary gate |

---

## 2. Memory input phải chia thành hai pha

Không nên dùng một object duy nhất cho cả retrieval và update. Nếu dùng chung `run.json` hoặc toàn bộ session metadata, rất dễ vô tình đưa oracle fields như `problem_names`, `root_cause_name`, hoặc ground-truth-derived fields vào memory.

Do đó cần tách thành:

1. **Retrieval input**: dùng trước khi solver chạy episode.
2. **Update input**: dùng sau khi solver chạy xong và evaluator đã sinh numeric metrics.

---

## 3. Pha trước episode: retrieval input

Retrieval input chỉ được chứa thông tin mà solver cũng được phép thấy trước khi bắt đầu điều tra.

Ví dụ:

```python
MemoryRetrievalInput = {
    "episode_id": "20260626-xxxxxx",
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

Retrieval input dùng để trả lời câu hỏi:

> Với incident public context như hiện tại, memory nào nên được đưa vào context cho solver?

Retrieval input không được chứa:

- injected fault;
- benchmark CSV `problem`;
- `problem_names`;
- `ground_truth.json`;
- correct faulty devices;
- correct root cause;
- failure injection params;
- evaluation score của episode hiện tại;
- bất kỳ dữ liệu nào chỉ có sau khi episode kết thúc.

### Vì sao phải chặt ở retrieval time?

Retrieval xảy ra trước khi solver hành động. Nếu retrieval input có `problem = dns_record_error`, thì chỉ riêng memory query cũng đã biết RCA label. Dù memory module không “cố tình” leak, vector search hoặc attribute filter cũng có thể retrieve đúng đáp án bằng oracle label.

Vì vậy, retrieval input phải là **public-only**.

---

## 4. Pha sau episode: update/evolve input

Sau khi solver hoàn thành episode và evaluator đã tính điểm, memory module có thể nhận thêm trajectory, submission và numeric feedback.

Ví dụ:

```python
MemoryUpdateInput = {
    "episode_id": "20260626-xxxxxx",
    "public_context": {
        "scenario_name": "ospf_enterprise_dhcp",
        "topology_class": "s",
        "scenario_params_public": {...},
        "topology_summary": {...},
        "task_description": "...",
        "protocols": ["ospf", "dhcp"],
        "services": ["dhcp", "dns"],
        "task_stage": "diagnosis"
    },
    "trajectory": [
        {
            "timestamp": "...",
            "phase": "diagnosis",
            "actor": "solver",
            "event_type": "tool_start",
            "tool_name": "show_ip_route",
            "content": "...",
            "status": "started"
        },
        {
            "timestamp": "...",
            "phase": "diagnosis",
            "actor": "solver",
            "event_type": "tool_result",
            "tool_name": "show_ip_route",
            "content": "...",
            "status": "ok"
        }
    ],
    "submission": {
        "is_anomaly": True,
        "faulty_devices": ["<device_A>"],
        "root_cause_name": ["dns_record_error"]
    },
    "numeric_feedback": {
        "detection_score": 1.0,
        "localization_f1": 1.0,
        "rca_f1": 1.0,
        "steps": 12,
        "tool_calls": 8,
        "tool_errors": 0,
        "in_tokens": 24000,
        "out_tokens": 3000
    }
}
```

Update input dùng để trả lời câu hỏi:

> Episode này sinh ra atomic procedural note nào, và note đó đáng staged, validated, rejected hay superseded?

Ở pha này, memory được phép biết solver đã submit gì và điểm số numeric là bao nhiêu. Nhưng memory vẫn không nên biết ground truth text trực tiếp.

---

## 5. Phân tích từng nhóm input

### 5.1. Episode identity và provenance

Ví dụ:

```python
{
    "episode_id": "...",
    "session_id": "...",
    "benchmark_file": "benchmark_selected.csv",
    "split": "evolution",
    "sequence_index": 12,
    "timestamp": "..."
}
```

Nên dùng để:

- audit memory sinh từ episode nào;
- reproducibility;
- tránh update trùng;
- snapshot và traceability;
- phân biệt evolution/test split.

Không nên dùng để:

- làm semantic retrieval key;
- tạo lesson kiểu “session này có lỗi X”;
- suy ra thứ tự benchmark để đoán đáp án.

Kết luận:

> `session_id` là provenance metadata, không phải nội dung semantic của memory.

---

### 5.2. Scenario metadata

Ví dụ:

```python
{
    "scenario_name": "dc_clos_bgp",
    "topology_class": "s",
    "scenario_params_public": {...},
    "lab_name": "..."
}
```

Đây là input rất hữu ích vì nhiều troubleshooting lesson phụ thuộc vào domain:

- BGP;
- OSPF;
- DHCP;
- DNS;
- VPN;
- P4/BMv2;
- data center Clos;
- enterprise routing;
- interdomain routing.

Memory có thể học lesson tốt như:

```text
Trong Clos BGP, nếu underlay reachability còn hoạt động nhưng traffic end-to-end fail,
hãy kiểm tra BGP route advertisement và next-hop reachability trước khi kết luận host failure.
```

Nhưng không được học lesson xấu như:

```text
Trong dc_clos_bgp size s, lỗi thường là host_crash.
```

Lesson thứ hai là benchmark memorization, không phải troubleshooting knowledge.

Kết luận:

> Scenario metadata nên dùng làm retrieval/filter attribute, nhưng persistent memory phải abstract thành diagnostic pattern.

---

### 5.3. Public task description

Task description là một trong các input quan trọng nhất ở retrieval time.

Nó có thể chứa:

- mô tả network;
- task objective;
- detection/localization/RCA yêu cầu gì;
- symptoms nếu benchmark cung cấp;
- instruction submit.

Vì solver cũng được thấy task description, memory được phép thấy nó.

Memory dùng task description để:

- tạo query retrieval;
- suy ra protocol/service hints;
- xác định task stage;
- tìm memories có pattern tương tự;
- chọn lessons phù hợp token budget.

Tuy nhiên, nếu task description chứa concrete identifiers như hostname/IP/interface, memory nên lưu bản redacted hoặc abstracted.

Ví dụ redaction:

```text
pc1, pc2, router1      → <host_A>, <host_B>, <router_A>
10.0.0.1               → <ip>
eth0                   → <interface>
00:11:22:33:44:55      → <mac>
```

Kết luận:

> Task description được dùng trực tiếp cho retrieval, nhưng khi lưu long-term memory nên dùng dạng redacted/abstracted.

---

### 5.4. Topology summary

Benchmark có thể cung cấp topology ở nhiều mức.

#### Mức 1: raw topology

```python
[
    ["pc1", "router1"],
    ["router1", "router2"],
    ["router2", "pc2"]
]
```

Raw topology có ích cho episode-local reasoning, nhưng nguy hiểm nếu lưu lâu dài vì dễ học thuộc benchmark instance.

#### Mức 2: normalized topology

```python
{
    "node_counts": {
        "host": 2,
        "router": 2,
        "switch": 0
    },
    "edge_count": 3,
    "protocols": ["bgp"],
    "shape": "linear_edge_to_edge"
}
```

Đây là dạng tốt hơn cho memory.

#### Mức 3: semantic topology class

```python
{
    "family": "interdomain_routing",
    "routing": "bgp",
    "shape": "two_as_edge_network",
    "scale": "small"
}
```

Đây là dạng lý tưởng cho retrieval và attribute matching.

Kết luận:

> Memory nên nhận topology summary đã normalize, không nên lưu raw hostname-level topology như long-term semantic content.

---

### 5.5. Protocol và service hints

Ví dụ:

```python
{
    "protocols": ["bgp", "ospf", "dhcp", "dns"],
    "services": ["nginx", "vpn"],
    "data_plane": ["p4", "bmv2"]
}
```

Đây là input rất mạnh cho structured retrieval.

Nó giúp memory module không phụ thuộc hoàn toàn vào vector similarity. Ví dụ, query về DHCP trong OSPF enterprise topology nên ưu tiên lessons liên quan DHCP/OSPF hơn BGP/P4.

Kết luận:

> Protocol/service hints nên được dùng như first-class attributes trong memory query và memory records.

---

### 5.6. Trajectory / observation stream

Trajectory là nguồn học chính cho update/evolve.

Trong repo hiện tại, trajectory chủ yếu nằm ở:

```text
results/<session_id>/messages.jsonl
```

Một trajectory event có thể được normalize thành:

```python
TraceEvent = {
    "timestamp": "...",
    "phase": "diagnosis",
    "actor": "solver",
    "event_type": "action | observation | result | error | final",
    "content": "...",
    "tool_name": "...",
    "status": "ok | error | started",
    "metadata": {...}
}
```

Trajectory có thể sinh ra nhiều atomic procedural notes:

| Từ trajectory | Memory có thể extract |
|---|---|
| solver kiểm tra gì trước | bước điều tra có thể tái sử dụng |
| quan sát network state | evidence pattern cần kiểm chứng |
| tool call lỗi | cảnh báo/cách tránh lỗi |
| bước nào dẫn tới chẩn đoán đúng | procedural note được validate bởi score |
| pattern lặp lại nhiều lần thành công | note có confidence và graph support cao hơn |

Ví dụ memory tốt:

```text
When diagnosing DHCP failures in an OSPF enterprise topology,
compare client lease state with DHCP server logs before assuming routing failure.
```

Ví dụ memory xấu:

```text
In session 20260626-abc123, pc2 was faulty.
```

Trajectory cần được tiền xử lý:

- bỏ các event từ submission-only phase nếu chỉ là format answer;
- bỏ hidden ground-truth text;
- redact IP/MAC/interface/hostname cụ thể khi lưu long-term;
- compact tool outputs dài;
- giữ lại observations có giá trị evidence;
- ưu tiên tool observations hơn self-talk của LLM.

Kết luận:

> Trajectory là input học quan trọng nhất, nhưng phải compact, filter và redact trước khi extract memory.

---

### 5.7. Solver submission

`submission.json` thường có dạng:

```python
{
    "is_anomaly": True,
    "faulty_devices": ["..."],
    "root_cause_name": ["..."]
}
```

Đây là output do solver tự submit, nên memory được phép thấy sau episode.

Nhưng ý nghĩa của submission phụ thuộc vào score.

Nếu success:

```python
submission = {"root_cause_name": ["dns_record_error"]}
rca_f1 = 1.0
```

Memory có thể xem đây là bằng chứng mạnh rằng trajectory đã dẫn tới RCA đúng.

Nếu fail:

```python
submission = {"root_cause_name": ["dns_record_error"]}
rca_f1 = 0.0
```

Memory không được suy ra đáp án đúng. Nó chỉ biết rằng kết luận này không được validate bởi evaluator.

Lesson hợp lệ từ failed episode:

```text
Do not conclude DNS record error from HTTP failure alone; verify resolver output first.
```

Lesson không hợp lệ:

```text
The correct root cause was not DNS record error and must be X.
```

Vì memory không được thấy X.

Kết luận:

> Submission là input hợp lệ sau episode, nhưng phải được diễn giải qua numeric feedback.

---

### 5.8. Numeric evaluation feedback

NIKA rule-based evaluator tạo các score như:

```python
{
    "detection_score": 1.0,
    "localization_accuracy": 1.0,
    "localization_precision": 1.0,
    "localization_recall": 1.0,
    "localization_f1": 1.0,
    "rca_accuracy": 1.0,
    "rca_precision": 1.0,
    "rca_recall": 1.0,
    "rca_f1": 1.0
}
```

Và trace stats:

```python
{
    "steps": 12,
    "tool_calls": 8,
    "tool_errors": 0,
    "in_tokens": 24000,
    "out_tokens": 3000
}
```

Đây là feedback hợp lệ nhất để evolve memory vì nó không tiết lộ text ground truth.

Score nên được dùng theo gate:

| Kết quả | Memory action |
|---|---|
| `detection_score=1`, `localization_f1=1`, `rca_f1=1` | validate atomic procedural notes |
| partial success | stage các note thận trọng, có evidence/avoid rõ |
| full failure | stage hoặc reject; không validate procedural claim |
| repeated full success cùng pattern | tăng confidence/validation count và graph support |
| contradiction với memory cũ | link `contradicts`, `refines`, hoặc supersede |

Không nên đưa vào memory generator prompt:

```text
The correct root cause was dns_record_error.
```

Nên đưa:

```text
This episode achieved detection_score=1.0,
localization_f1=1.0, rca_f1=1.0.
```

Kết luận:

> Numeric metrics là validation signal, không phải ground-truth explanation.

---

### 5.9. LLM judge feedback

NIKA có optional LLM-as-judge, có thể chấm:

- relevance;
- correctness;
- efficiency;
- clarity;
- final outcome;
- overall.

Không nên dùng LLM judge làm primary memory gate vì:

- kém deterministic hơn rule-based metric;
- khó reproduce;
- judge prompt thường có thể chứa ground truth;
- output judge có thể paraphrase ground truth;
- có bias theo văn phong của solver.

LLM judge chỉ nên dùng như auxiliary analysis signal, ví dụ:

- giải thích tại sao một episode tệ;
- phân tích clarity;
- hỗ trợ offline report.

Kết luận:

> LLM judge không nên là core input cho memory evolution.

---

## 6. Oracle/private fields phải bị cấm

NIKA có nhiều field hữu ích cho evaluator nhưng nguy hiểm với memory.

### 6.1. `problem` trong benchmark CSV

Ví dụ:

```csv
problem,scenario,topo_size
dns_record_error,ospf_enterprise_dhcp,s
host_crash,dc_clos_bgp,s
```

`problem` chính là fault được inject. Đây là oracle label.

Không được đưa vào:

- retrieval query;
- memory extraction prompt;
- memory content;
- attribute filters;
- embedding text.

---

### 6.2. `problem_names` trong session/run metadata

Khi fault được inject, session có thể lưu:

```python
problem_names = ["dns_record_error"]
```

Đây cũng là oracle label.

Không được đưa cho memory module, ngoại trừ dạng hash/audit private nếu thật sự cần reproducibility.

---

### 6.3. `root_cause_name` trong run metadata

Trong một số workflow, run metadata có thể có `root_cause_name` derived từ `problem_names`.

Field này tiện cho summary, nhưng với memory thì là leakage.

Không được dùng làm:

- memory label;
- query attribute;
- oracle-derived learning target;
- prompt input.

---

### 6.4. `ground_truth.json`

`ground_truth.json` chứa đáp án thật:

```python
{
    "is_anomaly": true,
    "faulty_devices": ["..."],
    "root_cause_name": ["..."]
}
```

Evaluator cần file này để tính score. Memory generator không nên thấy nó.

Memory chỉ nên nhận kết quả chấm dạng numeric:

```python
{
    "detection_score": 1.0,
    "localization_f1": 1.0,
    "rca_f1": 1.0
}
```

---

### 6.5. Failure injection params và verify result

Failure injection metadata có thể chứa:

- `faulty_devices`;
- `faulty_intf`;
- `intf_name`;
- `service_name`;
- `attacker_device`;
- `target_host`;
- `target_website`;
- `target_domain`;
- `p4_name`;
- `verify_result`;
- `requested_overrides`;
- `resolved_params`.

Các field này thường mô tả chính xác fault đã inject.

Ví dụ:

```python
{
    "problem_name": "host_ip_conflict",
    "injection_params": {
        "faulty_devices": ["pc2"],
        "intf_name": "eth0"
    }
}
```

Đây là oracle/private state. Không được dùng cho memory content.

---

### 6.6. Exact hostname/interface/IP mapping

Hostname, interface, IP không luôn là oracle vì solver có thể quan sát chúng trong lab. Nhưng nếu lưu lâu dài, chúng dễ biến thành benchmark memorization.

Ví dụ memory xấu:

```text
In dc_clos_bgp small, pc2 is usually faulty.
```

Memory tốt hơn:

```text
In Clos BGP incidents, if only one host pair loses reachability while underlay routes remain healthy,
compare host service state and BGP route export before localizing the failure.
```

Quy tắc:

- trong episode-local trace: có thể giữ raw identifiers để phân tích;
- trong persistent memory: nên redact/abstract;
- trong retrieved context: không đưa exact mapping kiểu `pc2 -> fault`;
- trong provenance nội bộ: có thể giữ session id hoặc source path.

---

## 7. Phân quyền input theo mức an toàn

| Input | Retrieval trước episode | Update sau episode | Persistent memory content | Ghi chú |
|---|---:|---:|---:|---|
| `session_id` | metadata only | metadata only | provenance only | không dùng semantic |
| `scenario_name` | có | có | attribute | không map trực tiếp sang đáp án |
| `topology_class` | có | có | attribute | nên abstract |
| `scenario_params_public` | có chọn lọc | có chọn lọc | attribute/metadata | lọc oracle params |
| topology raw | hạn chế | có chọn lọc | không nên raw | nên summarize |
| topology summary | có | có | có | input tốt |
| protocol/service hints | có | có | có | input rất tốt |
| task description | có | có | redacted | public signal |
| trajectory | không, vì chưa có | có | extract chọn lọc | compact + redact |
| tool outputs | không, vì chưa có | có | extract chọn lọc | ưu tiên evidence |
| tool errors | không, vì chưa có | có | có thể lưu caution/avoid note | hữu ích |
| submission | không, vì chưa có | có | có chọn lọc | diễn giải qua score |
| numeric metrics | không, vì chưa có | có | validation metadata | signal chính |
| LLM judge | không | optional | không nên core | kém deterministic |
| benchmark CSV `problem` | cấm | cấm | cấm | oracle |
| `problem_names` | cấm | cấm | cấm | oracle |
| `ground_truth.json` | cấm | evaluator only | cấm | oracle |
| injection params | cấm | cấm | cấm | oracle |
| verify result | cấm | cấm | cấm | oracle |

---

## 8. Contract đề xuất: `BenchmarkEpisode`

Một interface sạch nên tách public context, trajectory, submission, feedback và oracle reference.

```python
class BenchmarkEpisode:
    episode_id: str
    public_context: PublicContext
    trajectory: list[TraceEvent]
    submission: SolverSubmission | None
    feedback: NumericFeedback | None
    private_oracle_ref: OracleRef | None
```

`private_oracle_ref` chỉ để evaluator/audit, không bao giờ đưa vào memory generator hoặc retrieval.

### 8.1. `PublicContext`

```python
class PublicContext:
    scenario_name: str
    topology_class: str | None
    scenario_params_public: dict
    topology_summary: dict
    task_description: str
    protocols: list[str]
    services: list[str]
    task_schema: dict
```

### 8.2. `TraceEvent`

```python
class TraceEvent:
    timestamp: str
    phase: str
    event_type: str
    actor: str
    content: str
    tool_name: str | None
    status: str | None
    redaction_level: str
```

### 8.3. `SolverSubmission`

```python
class SolverSubmission:
    is_anomaly: bool
    faulty_devices: list[str]
    root_cause_name: list[str]
```

### 8.4. `NumericFeedback`

```python
class NumericFeedback:
    detection_score: float
    localization_accuracy: float | None
    localization_precision: float | None
    localization_recall: float | None
    localization_f1: float
    rca_accuracy: float | None
    rca_precision: float | None
    rca_recall: float | None
    rca_f1: float
    steps: int | None
    tool_calls: int | None
    tool_errors: int | None
    in_tokens: int | None
    out_tokens: int | None
```

### 8.5. `OracleRef`

```python
class OracleRef:
    ground_truth_path: str
    problem_names_hash: str
    injection_id: str | None
```

`OracleRef` không chứa:

- ground truth text;
- raw problem name;
- correct faulty devices;
- correct RCA label.

Nếu cần audit, chỉ lưu reference hoặc hash, không đưa vào LLM.

---

## 9. Memory retrieval schema đề xuất

```python
MemoryRetrievalInput = {
    "episode_id": "...",
    "query_text": "...",             # redacted task description
    "scenario": {
        "name": "ospf_enterprise_dhcp",
        "topology_class": "s",
        "family": "enterprise",
        "protocols": ["ospf", "dhcp"],
        "services": ["dhcp", "dns"]
    },
    "task": {
        "stage": "diagnosis",
        "schema": {
            "is_anomaly": "bool",
            "faulty_devices": "list[str]",
            "root_cause_name": "list[str]"
        }
    },
    "limits": {
        "candidate_limit": 20,
        "top_k": 5,
        "token_budget": 1500
    }
}
```

Retrieval output nên là:

```python
MemoryRetrievalOutput = {
    "memories": [
        {
            "memory_id": "...",
            "content": "...",
            "confidence": 0.87,
            "attributes": {...},
            "score": 0.73
        }
    ],
    "audit": {
        "candidate_count": 20,
        "selected_count": 5,
        "token_estimate": 1300
    }
}
```

---

## 10. Memory update schema đề xuất

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
        "excluded_phases": ["submission_format_only"],
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

Update output nên là:

```python
MemoryUpdateOutput = {
    "status": "completed",
    "candidate_count": 4,
    "accepted_count": 2,
    "staged_count": 0,
    "validated_count": 2,
    "superseded_count": 0,
    "rejected_count": 2,
    "memory_ids": ["..."],
    "snapshot_path": "..."
}
```

---

## 11. Sanitizer đề xuất

### 11.1. Retrieval sanitizer

Pseudocode:

```python
def make_retrieval_input(run_meta: dict) -> MemoryRetrievalInput:
    public = {}
    public["episode_id"] = run_meta["session_id"]
    public["scenario_name"] = run_meta.get("scenario_name", "")
    public["topology_class"] = run_meta.get("scenario_topo_size", "")
    public["task_description"] = redact(run_meta.get("task_description", ""))
    public["scenario_params_public"] = strip_oracle_params(
        run_meta.get("scenario_params", {})
    )
    public["topology_summary"] = summarize_topology(
        run_meta.get("topology", [])
    )
    public["protocols"] = infer_protocols(public)
    public["services"] = infer_services(public)
    public["task_stage"] = "diagnosis"
    return MemoryRetrievalInput(**public)
```

Phải drop:

```python
FORBIDDEN_KEYS = {
    "problem",
    "problem_names",
    "root_cause_name",
    "ground_truth",
    "ground_truth_path",
    "faulty_devices",
    "failure_injections",
    "injection_params",
    "verify_result",
    "requested_overrides",
    "resolved_params",
}
```

### 11.2. Update sanitizer

Pseudocode:

```python
def make_update_input(
    run_meta: dict,
    messages_jsonl: list[dict],
    submission: dict | None,
    eval_metrics: dict,
) -> MemoryUpdateInput:
    public_context = make_public_context(run_meta)
    trajectory = compact_and_redact_trajectory(messages_jsonl)
    safe_submission = normalize_submission(submission)
    feedback = numeric_feedback_only(eval_metrics)

    return MemoryUpdateInput(
        episode_id=run_meta["session_id"],
        public_context=public_context,
        trajectory=trajectory,
        submission=safe_submission,
        feedback=feedback,
    )
```

Update sanitizer vẫn không đọc `ground_truth.json` để tạo prompt memory.

Evaluator có thể đọc `ground_truth.json` để sinh `eval_metrics.json`, nhưng memory chỉ nhận `eval_metrics.json`.

---

## 12. Redaction rules

Persistent memory nên tránh exact identifiers. Redaction đề xuất:

| Pattern | Replace with |
|---|---|
| IPv4/IPv6 address | `<ip>` |
| MAC address | `<mac>` |
| interface name như `eth0`, `ens3`, `r1-eth0` | `<interface>` |
| host/router/switch concrete names | `<device_A>`, `<router_A>`, `<host_A>` |
| session id | `<session_id>` |
| container id | `<container_id>` |
| benchmark problem id khi xuất hiện ngoài submission | `<problem_id>` |

Không phải mọi identifier đều phải bị xoá khỏi trajectory tạm thời. Nhưng khi viết long-term memory content, nên abstract hóa.

---

## 13. Memory nên học loại tri thức gì?

Memory module nên học **diagnostic policy**, không học benchmark answers.

### 13.1. Evidence-pattern note

Ví dụ:

```text
In OSPF enterprise incidents, DHCP client failures can appear even when OSPF adjacency is healthy.
```

Nguồn:

- tool observations;
- redacted trajectory;
- failed hoặc successful episodes.

Status ban đầu:

- thường là `staged`;
- có thể validate nếu lặp lại hoặc episode full success.

### 13.2. Caution/avoid note

Ví dụ:

```text
Do not conclude DNS record error solely from HTTP failure; verify name resolution output first.
```

Nguồn:

- failed episode;
- partial score;
- tool errors;
- bad reasoning path.

Status ban đầu:

- thường là `staged`;
- validate nếu cùng lỗi tránh được trong later success.

### 13.3. Validated procedural note

Ví dụ:

```text
For BGP reachability failures, compare route advertisement and next-hop reachability before localizing the host.
```

Nguồn:

- full successful episode;
- trajectory có evidence rõ.

Status:

- `validated` nếu detection/localization/RCA đều đạt full score.

### 13.4. Highly corroborated graph-supported note

Ví dụ:

```text
When diagnosing DHCP service issues, check lease state, server process health,
and client resolver config before RCA submission.
```

Nguồn:

- repeated validated atomic notes;
- nhiều episode cùng support;
- các link `supports`, `same_pattern`, hoặc `refines` trong memory graph.

Status:

- vẫn là atomic note;
- confidence, validation count và graph support tăng lên sau nhiều lần corroboration.

---

## 14. Độ tin cậy của từng loại signal

| Signal | Độ tin cậy | Giải thích |
|---|---:|---|
| Tool observation | cao | đến từ environment, ít hallucination hơn self-talk |
| Full numeric success | rất cao | xác nhận trajectory + submission dẫn tới đáp án đúng |
| Repeated success cùng pattern | rất cao | tăng confidence và graph support |
| Partial localization/RCA | trung bình | có thể đúng một phần |
| Tool error | trung bình | hữu ích cho caution/avoid notes |
| Failed episode | thấp/trung bình | biết cái gì không được validate, không biết đáp án đúng |
| Raw LLM reasoning | thấp/trung bình | có thể hallucinate |
| LLM judge | trung bình/thấp | non-deterministic, có nguy cơ leak |
| Ground truth | oracle nhưng không hợp lệ | chỉ evaluator dùng |

Nguyên tắc:

> Memory nên ưu tiên observed evidence + successful outcome, không ưu tiên self-talk.

---

## 15. Online evolution phải theo thứ tự

Nếu benchmark dùng memory evolution online, thứ tự episode rất quan trọng.

Đúng:

```text
episode 1 retrieval from memory_so_far
episode 1 run solver
episode 1 evaluate
episode 1 update memory

episode 2 retrieval from memory_so_far
episode 2 run solver
episode 2 evaluate
episode 2 update memory
```

Sai nếu muốn đánh giá online memory:

```text
episode 1, 2, 3 run parallel
all update the same memory bank
```

Lý do:

- episode có thể học từ future episode;
- update order không deterministic;
- score khó reproduce;
- memory bank trở thành shared mutable state không kiểm soát.

Quy tắc:

- `memory_mode=evolve`: nên chạy sequential.
- `memory_mode=read`: có thể chạy parallel nếu memory bank frozen.
- mỗi experiment/seed nên có memory bank riêng.

---

## 16. Evaluation metrics cho memory module

Tài liệu này không định nghĩa biến thể memory so sánh bắt buộc. Memory module của dự án
nên được đánh giá như một implementation duy nhất dựa trên:

- MemInsight-style context retrieval;
- LightMem-style offline update/consolidation;
- A-Mem-style atomic procedural note và memory graph.

Các metric nên theo dõi:

- detection score;
- localization F1;
- RCA F1;
- task success rate;
- steps;
- tool calls;
- tool errors;
- token usage;
- memory retrieval precision qualitative;
- memory update acceptance rate;
- memory contradiction/supersede rate.

---

## 17. Recommended benchmark-to-memory API

Nếu thiết kế API độc lập với agent implementation, nên có 4 hàm:

```python
class BenchmarkMemoryAdapter:
    def build_retrieval_input(run_meta: dict) -> MemoryRetrievalInput:
        ...

    def build_update_input(
        run_meta: dict,
        messages_jsonl: Iterable[dict],
        submission: dict | None,
        eval_metrics: dict,
    ) -> MemoryUpdateInput:
        ...

    def assert_no_oracle_leakage(payload: dict) -> None:
        ...

    def snapshot_episode_input(payload: dict, output_path: str) -> None:
        ...
```

Memory module bên dưới chỉ cần biết:

```python
retrieve(MemoryRetrievalInput) -> list[Memory]
update(MemoryUpdateInput) -> MemoryUpdateReport
```

Nó không nên tự đọc bừa `run.json`, `ground_truth.json`, session store, hoặc benchmark CSV. Adapter chịu trách nhiệm sanitize.

---

## 18. Checklist chống leakage

Trước khi đưa bất kỳ payload nào vào memory LLM hoặc embedding text, kiểm tra:

- [ ] Payload không có key `problem`.
- [ ] Payload không có key `problem_names`.
- [ ] Payload không có key `ground_truth`.
- [ ] Payload không có raw `ground_truth.json`.
- [ ] Payload không có `root_cause_name` derived từ run metadata.
- [ ] Payload không có injection params.
- [ ] Payload không có verify result.
- [ ] Payload không có correct faulty devices từ evaluator.
- [ ] Payload không chứa benchmark CSV row đầy đủ.
- [ ] Hostname/IP/interface đã được redact nếu lưu long-term.
- [ ] Submission được đánh dấu rõ là solver output, không phải ground truth.
- [ ] Metrics chỉ là numeric feedback.
- [ ] Retrieval input không chứa bất kỳ thông tin post-episode nào.
- [ ] Evolution mode chạy sequential hoặc memory bank frozen.

---

## 19. Kết luận

Input hợp lệ cho memory module từ NIKA nên là:

```text
public episode context
+ redacted solver trajectory
+ solver submission
+ numeric evaluation feedback
```

Cụ thể:

```text
Before episode:
  scenario name
  topology class
  public topology summary
  protocol/service hints
  task description
  task stage/schema

After episode:
  redacted trajectory
  solver submission
  rule-based numeric metrics
  trace stats
```

Không hợp lệ:

```text
benchmark problem label
problem_names
ground_truth.json
root_cause_name from run metadata
failure injection params
verify result
exact benchmark mapping scenario → answer
```

Thiết kế này giữ NIKA đúng vai trò benchmark/evaluator, còn memory module là một implementation độc lập học từ kinh nghiệm quan sát được, thay vì học thuộc đáp án benchmark.
