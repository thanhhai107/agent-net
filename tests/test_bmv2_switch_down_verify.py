from nika.orchestrator.problems.network_node_error.swicth_router_failure import (
    _simple_switch_has_live_process,
)


def test_simple_switch_live_process_detected_from_ps() -> None:
    ps_output = "36 S simple_switch simple_switch -i 1@eth0 program.json"

    assert _simple_switch_has_live_process(ps_output) is True


def test_simple_switch_zombie_process_is_not_live() -> None:
    ps_output = "36 Z simple_switch [simple_switch] <defunct>"
    pgrep_output = "36 [simple_switch] <defunct>"

    assert _simple_switch_has_live_process(ps_output, pgrep_output) is False


def test_simple_switch_pgrep_defunct_fallback_is_not_live() -> None:
    assert (
        _simple_switch_has_live_process(
            ps_output="",
            pgrep_output="36 [simple_switch] <defunct>",
        )
        is False
    )


def test_simple_switch_pgrep_live_fallback_is_live() -> None:
    assert (
        _simple_switch_has_live_process(
            ps_output="",
            pgrep_output="36 simple_switch -i 1@eth0 program.json",
        )
        is True
    )
