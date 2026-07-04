from __future__ import annotations

import json
from pathlib import Path

from nika.workflows.benchmark import run as benchmark_run


def test_benchmark_yaml_resume_skips_finished_indices(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    benchmark_file = tmp_path / "benchmark_resume.yaml"
    benchmark_file.write_text(
        """
cases:
- scenario: simple_bgp
  problem: no_fault
  inject: {}
- scenario: simple_bgp
  problem: link_down
  inject:
    host_name: pc1
    intf_name: eth0
- scenario: simple_bgp
  problem: link_flap
  inject:
    host_name: pc1
    intf_name: eth0
""",
        encoding="utf-8",
    )
    old_session = tmp_path / "old-session"
    old_session.mkdir()
    (old_session / "run.json").write_text(
        json.dumps({"status": "finished", "benchmark_index": 0}),
        encoding="utf-8",
    )
    (old_session / "eval_metrics.json").write_text("{}", encoding="utf-8")

    calls: list[int | None] = []

    def fake_run_single_benchmark(**kwargs):
        calls.append(kwargs.get("benchmark_index"))
        return f"session-{kwargs.get('benchmark_index')}"

    monkeypatch.setattr(benchmark_run, "validate_agent_extensions", lambda *_: None)
    monkeypatch.setattr(benchmark_run, "run_single_benchmark", fake_run_single_benchmark)

    benchmark_run.run_benchmark_from_yaml(
        benchmark_file=str(benchmark_file),
        agent_type="mock",
        llm_backend="mock",
        model="mock",
        max_steps=1,
        result_root=tmp_path,
        resume=True,
    )

    assert calls == [1, 2]
    output = capsys.readouterr().out
    assert "benchmark_progress total=3 completed=1 failed=0 skipped=1" in output
    assert "benchmark_skip index=1/3" in output
    assert "scenario=simple_bgp topo_size=- problem=no_fault" in output
    assert "benchmark_start index=2/3 scenario=simple_bgp topo_size=- problem=link_down inject_host_name=pc1 inject_intf_name=eth0" in output
    assert "benchmark_summary total=3 completed=3 failed=0 skipped=1" in output
