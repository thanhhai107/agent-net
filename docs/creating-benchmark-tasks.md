# Creating Benchmark Tasks

This guide shows how to add a new NIKA benchmark task using the existing APIs for network instantiation, failure injection, traffic generation, and benchmark execution.

## Task Model

A benchmark case combines:

- a network scenario from `src/nika/net_env/`
- one injectable problem from `src/nika/orchestrator/problems/`
- explicit injection parameters
- optional traffic generated before or during troubleshooting
- an agent run and evaluation output under `results/{session_id}/`

The standard pipeline is:

```shell
nika env run <scenario> [-s s|m|l]
nika failure inject <problem> --set key=value ...
nika traffic run <type> ...
nika agent run -a <agent> ...
nika session close -y
nika eval metrics
```

For benchmark automation, `nika benchmark run` performs env deploy, fault injection, agent run, close, and eval for each case.

## Add A Network Scenario

Network environments are Kathara labs wrapped by `NetworkEnvBase`.

1. Add the lab under `src/nika/net_env/<domain>/<scenario>/`.
2. Implement a class that sets `LAB_NAME`, builds `self.lab`, sets `self.name`, `self.desc`, and declares useful host lists through `load_machines()`.
3. If the scenario has sizes, expose `TOPO_SIZE = ["s", "m", "l"]` and accept `topo_size` in `__init__`.
4. Register the class in `src/nika/net_env/net_env_pool.py`.

Minimal shape:

```python
from Kathara.model.Lab import Lab
from Kathara.manager.Kathara import Kathara

from nika.net_env.base import NetworkEnvBase


class MyScenario(NetworkEnvBase):
    LAB_NAME = "my_scenario"
    TOPO_SIZE = ["s", "m", "l"]  # omit for fixed-size labs

    def __init__(self, topo_size: str = "s"):
        super().__init__()
        self.name = self.LAB_NAME
        self.desc = "Short operator-facing description."
        self.instance = Kathara.get_instance()
        self.lab = Lab(self.name)

        pc1 = self.lab.new_machine("pc1", image="kathara/nika-base")
        pc2 = self.lab.new_machine("pc2", image="kathara/nika-base")
        self.lab.connect_machine_to_link(pc1.name, "A")
        self.lab.connect_machine_to_link(pc2.name, "A")

        self.load_machines()
```

Registration:

```python
from nika.net_env.example.my_scenario.lab import MyScenario

_NET_ENVS = {
    # ...
    MyScenario.LAB_NAME: MyScenario,
}
```

Verify discovery and deployment:

```shell
uv run nika env list
uv run nika env run my_scenario -s s
uv run nika session inspect
uv run nika session close -y
```

## Add An Injectable Problem

Problems live under `src/nika/orchestrator/problems/<category>/`. They are discovered automatically when their concrete task classes subclass `TaskBase` through `DetectionTask`, `LocalizationTask`, or `RCATask`.

Use a shared base class for the fault logic, then expose three task classes.

```python
from pydantic import BaseModel, Field

from nika.generator.fault.injector_base import FaultInjectorBase
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.problems.problem_base import (
    ProblemMeta,
    RootCauseCategory,
    TaskDescription,
    TaskLevel,
    build_verify_result,
)
from nika.orchestrator.tasks.detection import DetectionTask
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask


class MyFaultParams(BaseModel):
    host_name: str = Field(description="Target host.")
    intf_name: str = Field(default="eth0", description="Target interface.")


class MyFaultBase:
    root_cause_category = RootCauseCategory.LINK_FAILURE
    root_cause_name = "my_fault"
    TAGS = ["link"]
    Params = MyFaultParams

    symptom_desc = "Users report intermittent connectivity."

    def __init__(self, scenario_name: str | None, **kwargs):
        super().__init__()
        self.net_env = get_net_env_instance(scenario_name, **kwargs)
        self.injector = FaultInjectorBase(lab_name=self.net_env.lab.name)
        self.faulty_devices: list[str] = []

    def inject_fault(self, params: MyFaultParams):
        self.faulty_devices = [params.host_name]
        self.injector.inject_intf_down(params.host_name, params.intf_name)

    def verify_fault(self, params: MyFaultParams) -> dict:
        return build_verify_result(
            root_cause_name=self.root_cause_name,
            faulty_devices=self.faulty_devices,
            verified=True,
            details={"host": params.host_name, "intf": params.intf_name},
        )


class MyFaultDetection(MyFaultBase, DetectionTask):
    META = ProblemMeta(
        root_cause_category=MyFaultBase.root_cause_category,
        root_cause_name=MyFaultBase.root_cause_name,
        task_level=TaskLevel.DETECTION,
        description=TaskDescription.DETECTION,
    )


class MyFaultLocalization(MyFaultBase, LocalizationTask):
    META = ProblemMeta(
        root_cause_category=MyFaultBase.root_cause_category,
        root_cause_name=MyFaultBase.root_cause_name,
        task_level=TaskLevel.LOCALIZATION,
        description=TaskDescription.LOCALIZATION,
    )


class MyFaultRCA(MyFaultBase, RCATask):
    META = ProblemMeta(
        root_cause_category=MyFaultBase.root_cause_category,
        root_cause_name=MyFaultBase.root_cause_name,
        task_level=TaskLevel.RCA,
        description=TaskDescription.RCA,
    )
```

Notes:

- `Params` must be a Pydantic model. `nika failure describe` and benchmark YAML validation use it as the injection schema.
- `inject_fault()` should mutate only the selected lab instance.
- `verify_fault()` must prove the fault is active. Failed verification marks the injection as failed and stops the run.
- Set `faulty_devices` and other task attributes before ground truth is written.

Verify the problem:

```shell
uv run nika failure list
uv run nika failure describe my_fault
uv run nika env run my_scenario -s s
uv run nika failure inject my_fault --set host_name=pc1 --set intf_name=eth0
uv run nika failure ps
```

## Generate Traffic

Use the built-in traffic generators when a task needs load or baseline activity.

OD-matrix iperf3 traffic:

```python
import asyncio

from nika.generator.traffic.od_flows import ODFLowGenerator


async def run_traffic(lab_name: str):
    generator = ODFLowGenerator(lab_name=lab_name)
    return await generator.astart_generate_traffic(
        {"pc1": {"pc2": 20}},
        interval=60,
        unit="M",
        udp=True,
    )


asyncio.run(run_traffic("my_scenario__instance"))
```

CLI equivalent:

```shell
nika traffic run od --all-to-host pc2 --mbps 20 --interval 60
nika traffic run od --mesh-mbps 5 --interval 300 --background
```

Web browsing traffic requires the scenario to define `web_urls` and web servers discoverable by `load_machines()`:

```shell
nika traffic run web --pages-min 2 --pages-max 5 --no-loop
```

For faults that create traffic as the root cause, keep that logic inside the problem class. For background or validation traffic, prefer the traffic CLI or generator APIs.

## Add Benchmark Cases

Benchmark YAML rows use the same names and injection parameters as the CLI:

```yaml
cases:
  - scenario: my_scenario
    topo_size: s
    problem: my_fault
    inject:
      host_name: pc1
      intf_name: eth0
```

Run a single case:

```shell
uv run nika benchmark run my_scenario --problem my_fault -s s \
  --set host_name=pc1 --set intf_name=eth0 \
  -a mock -m mock-v1
```

Run a YAML file:

```shell
uv run nika benchmark run --config benchmark/my_cases.yaml \
  --result_dir results/my_cases \
  -a mock -m mock-v1
```

## Validation Checklist

- `uv run nika env list` shows the scenario.
- `uv run nika env run <scenario>` deploys and creates a session.
- `uv run nika failure describe <problem>` shows the expected schema.
- `uv run nika failure inject <problem> --set ...` verifies successfully.
- `uv run nika benchmark run ... -a mock -m mock-v1` completes without external LLM credentials.
- The session directory contains `ground_truth.json`, `run.json`, `events.jsonl`, and evaluation artifacts.
