"""Streamlit dashboard for NIKA troubleshooting sessions."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

from nika.visualization.data import (
    discover_sessions,
    fault_endpoints,
    faulty_devices,
    load_session_bundle,
    parse_topology,
    replay_steps,
    timeline_rows,
)
from nika.visualization.topology import render_topology_svg


st.set_page_config(
    page_title="NIKA · Network Arena",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #e8eef7;
        --muted: #8d9bb0;
        --panel: rgba(13, 23, 38, .78);
        --panel-strong: #101c2d;
        --line: rgba(142, 164, 190, .17);
        --cyan: #37d4c7;
        --blue: #63a4ff;
        --red: #ff647c;
        --amber: #ffb454;
        --violet: #a78bfa;
      }

      .stApp {
        background:
          radial-gradient(circle at 12% 4%, rgba(25, 101, 116, .18), transparent 24rem),
          radial-gradient(circle at 92% 12%, rgba(42, 78, 145, .16), transparent 26rem),
          #08111e;
        color: var(--ink);
      }
      .block-container {max-width: 1480px; padding: 1.7rem 2rem 4rem;}
      section[data-testid="stSidebar"] {
        background: rgba(7, 15, 27, .96);
        border-right: 1px solid var(--line);
      }
      section[data-testid="stSidebar"] .block-container {padding: 1.35rem 1.15rem;}
      h1, h2, h3 {letter-spacing: -.025em;}
      h1 {font-size: clamp(2rem, 3vw, 3.15rem) !important; line-height: 1.05 !important;}
      h2 {font-size: 1.28rem !important;}
      h3 {font-size: 1.02rem !important; color: #cdd8e7 !important;}

      [data-testid="stMetric"] {
        min-height: 112px;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 17px 18px;
        background: linear-gradient(145deg, rgba(18, 33, 52, .92), rgba(9, 20, 35, .82));
        box-shadow: 0 12px 35px rgba(0, 0, 0, .15);
      }
      [data-testid="stMetricLabel"] {color: var(--muted); font-size: .78rem;}
      [data-testid="stMetricValue"] {color: var(--ink); font-weight: 700;}

      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .35rem; background: rgba(10, 20, 34, .65); border: 1px solid var(--line);
        border-radius: 14px; padding: .32rem; margin: .7rem 0 1.2rem;
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 42px; border-radius: 10px; padding: 0 1.2rem;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(55, 212, 199, .12); color: #8bf3e9;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
      }
      [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 13px;
        background: rgba(12, 23, 39, .58);
      }
      .stButton > button, .stDownloadButton > button {
        border-radius: 11px; border: 1px solid rgba(55, 212, 199, .28);
        background: rgba(55, 212, 199, .08);
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--cyan); color: #a7fff7;
      }

      .nika-brand {display:flex; align-items:center; gap:.75rem; margin:.2rem 0 1.6rem;}
      .nika-mark {
        width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
        color:#07131f; background:linear-gradient(135deg,#7cf4e9,#5d9fff);
        font-size:1.15rem; font-weight:900; box-shadow:0 0 24px rgba(55,212,199,.2);
      }
      .nika-brand-title {font-size:1.05rem; font-weight:800; letter-spacing:.08em;}
      .nika-brand-sub {font-size:.72rem; color:var(--muted); letter-spacing:.04em;}

      .eyebrow {
        color:#6fe7dc; font-size:.74rem; font-weight:800; letter-spacing:.14em;
        text-transform:uppercase; margin-bottom:.55rem;
      }
      .hero-sub {color:var(--muted); font-size:.98rem; margin-top:.4rem;}
      .hero-meta {display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
      .chip {
        display:inline-flex; align-items:center; gap:.38rem; padding:.34rem .65rem;
        border:1px solid var(--line); border-radius:999px; color:#b9c7d9;
        background:rgba(13,25,42,.7); font-size:.76rem;
      }
      .dot {width:7px;height:7px;border-radius:50%;background:#7f8da1;}
      .dot.running {background:#41dfb2;box-shadow:0 0 12px rgba(65,223,178,.65);}
      .dot.finished {background:#6da8ff;}

      .section-label {
        color:#dbe6f4; font-weight:750; font-size:1rem; margin:.25rem 0 .9rem;
      }
      .glass-card {
        height:100%; border:1px solid var(--line); border-radius:16px;
        padding:1.15rem 1.2rem; background:var(--panel);
        box-shadow:0 14px 35px rgba(0,0,0,.12);
      }
      .card-kicker {
        color:var(--muted); font-size:.7rem; font-weight:800; letter-spacing:.11em;
        text-transform:uppercase; margin-bottom:.85rem;
      }
      .kv-grid {display:grid;grid-template-columns:minmax(95px,.72fr) 1.6fr;gap:.64rem .85rem;}
      .kv-key {color:var(--muted);font-size:.8rem;}
      .kv-value {color:#dce7f5;font-size:.82rem;font-weight:600;overflow-wrap:anywhere;}
      .diagnosis-title {font-size:1.08rem;font-weight:800;margin-bottom:.85rem;}
      .diagnosis-row {display:flex;align-items:flex-start;gap:.7rem;margin:.62rem 0;}
      .diagnosis-icon {
        width:25px;height:25px;flex:0 0 25px;display:grid;place-items:center;
        border-radius:8px;font-size:.72rem;font-weight:900;
      }
      .diagnosis-label {color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;}
      .diagnosis-value {color:#e3edf9;font-size:.86rem;font-weight:650;margin-top:.12rem;}
      .match-banner {
        margin-top:1rem;padding:.72rem .85rem;border-radius:11px;font-size:.8rem;font-weight:700;
      }
      .match-good {background:rgba(54,211,153,.1);color:#75edc0;border:1px solid rgba(54,211,153,.2);}
      .match-bad {background:rgba(255,100,124,.09);color:#ff9aac;border:1px solid rgba(255,100,124,.2);}
      .empty-state {
        padding:2.8rem 1rem;text-align:center;border:1px dashed rgba(142,164,190,.25);
        border-radius:16px;color:var(--muted);background:rgba(10,21,36,.35);
      }
      .legend-row {display:flex;flex-wrap:wrap;gap:.55rem;margin:.25rem 0 1rem;}
      .legend-chip {
        display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .62rem;
        border:1px solid var(--line);border-radius:9px;color:#aebed2;font-size:.74rem;
        background:rgba(12,23,39,.65);
      }
      .legend-swatch {width:9px;height:9px;border-radius:50%;}
      .sidebar-note {
        margin-top:1.4rem;padding:1rem;border:1px solid var(--line);border-radius:13px;
        background:rgba(13,25,42,.58);color:var(--muted);font-size:.77rem;line-height:1.55;
      }
      .replay-card {
        border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.25rem;
        background:linear-gradient(145deg,rgba(18,33,52,.9),rgba(9,20,35,.82));
        min-height:180px;
      }
      .replay-top {display:flex;align-items:center;justify-content:space-between;gap:1rem;}
      .replay-kind {
        color:#71e9de;font-size:.69rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;
      }
      .replay-time {color:var(--muted);font-size:.72rem;}
      .replay-title {font-size:1.25rem;font-weight:800;margin:.55rem 0 .25rem;}
      .replay-agent {color:#91a4bb;font-size:.82rem;}
      .device-chip {
        display:inline-flex;padding:.28rem .55rem;margin:.7rem .3rem 0 0;border-radius:8px;
        background:rgba(55,212,199,.1);border:1px solid rgba(55,212,199,.22);
        color:#83eee4;font-size:.72rem;font-weight:700;
      }
      .step-track {display:flex;gap:4px;margin:1rem 0 .4rem;overflow:hidden;}
      .step-segment {height:5px;flex:1;border-radius:99px;background:#22344a;}
      .step-segment.done {background:#438ca1;}
      .step-segment.active {background:#62f2e5;box-shadow:0 0 12px rgba(98,242,229,.5);}
      @media (max-width: 800px) {
        .block-container {padding:1.15rem .9rem 3rem;}
        .kv-grid {grid-template-columns:1fr;}
        .kv-key {margin-top:.3rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe(value: object, fallback: str = "—") -> str:
    if value is None or value == "" or value == []:
        return fallback
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return html.escape(str(value))


def _fmt_score(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_time(value: object) -> str:
    if not value:
        return "—"
    raw = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).strftime("%b %d, %Y · %H:%M:%S")
    except ValueError:
        return str(value)


def _session_label(session: dict[str, Any]) -> str:
    status = "●" if session.get("status") == "running" else "○"
    scenario = session.get("scenario_name") or "unknown"
    tier = session.get("scenario_topo_size")
    topology = f"{scenario}/{tier}" if tier else scenario
    return f"{status}  {topology}  ·  {session.get('session_id', '')}"


def _card(title: str, values: list[tuple[str, object]]) -> None:
    rows = "".join(
        f'<div class="kv-key">{html.escape(key)}</div><div class="kv-value">{_safe(value)}</div>'
        for key, value in values
    )
    st.markdown(
        f'<div class="glass-card"><div class="card-kicker">{html.escape(title)}</div>'
        f'<div class="kv-grid">{rows}</div></div>',
        unsafe_allow_html=True,
    )


def _diagnosis_card(title: str, payload: dict[str, Any], accent: str) -> None:
    anomaly = payload.get("is_anomaly")
    anomaly_text = "Anomaly detected" if anomaly else "No anomaly"
    devices = payload.get("faulty_devices") or []
    causes = payload.get("root_cause_name") or []
    st.markdown(
        f"""
        <div class="glass-card">
          <div class="card-kicker">{html.escape(title)}</div>
          <div class="diagnosis-title">{anomaly_text if payload else "Not available"}</div>
          <div class="diagnosis-row">
            <div class="diagnosis-icon" style="background:{accent}20;color:{accent}">N</div>
            <div><div class="diagnosis-label">Faulty devices</div>
            <div class="diagnosis-value">{_safe(devices)}</div></div>
          </div>
          <div class="diagnosis-row">
            <div class="diagnosis-icon" style="background:{accent}20;color:{accent}">R</div>
            <div><div class="diagnosis-label">Root cause</div>
            <div class="diagnosis-value">{_safe(causes)}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <div style="font-size:2rem;margin-bottom:.6rem">◇</div>
          <div style="color:#d8e4f2;font-weight:750">No troubleshooting sessions yet</div>
          <div style="margin-top:.4rem">Start one with <code>nika env run &lt;scenario&gt;</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _move_replay(key: str, delta: int, maximum: int) -> None:
    current = int(st.session_state.get(key, 0))
    st.session_state[key] = min(maximum, max(0, current + delta))


def _set_replay(key: str, value: int) -> None:
    st.session_state[key] = value


def _render_dashboard() -> None:
    sessions = discover_sessions()
    selected_id: str | None = None

    with st.sidebar:
        st.markdown(
            """
            <div class="nika-brand">
              <div class="nika-mark">N</div>
              <div><div class="nika-brand-title">NIKA</div>
              <div class="nika-brand-sub">NETWORK ARENA</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not sessions:
            st.caption("No sessions found")
        else:
            session_ids = [str(item["session_id"]) for item in sessions]
            requested = os.environ.get("NIKA_VISUALIZE_SESSION_ID")
            default_index = session_ids.index(requested) if requested in session_ids else 0
            selected_id = st.selectbox(
                "Active session",
                session_ids,
                index=default_index,
                format_func=lambda session_id: _session_label(
                    next(item for item in sessions if item["session_id"] == session_id)
                ),
            )
            if st.button("↻  Refresh artifacts", width="stretch"):
                st.rerun()
            st.markdown(
                """
                <div class="sidebar-note">
                  <b style="color:#cbd8e8">Live artifact view</b><br>
                  This console reads session files directly. Refresh while an agent is running
                  to see new events, tool calls, and submissions.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not sessions:
        st.markdown('<div class="eyebrow">Network troubleshooting observatory</div>', unsafe_allow_html=True)
        st.title("See the network think.")
        _render_empty()
        return

    assert selected_id is not None
    bundle = load_session_bundle(selected_id)
    meta = bundle.meta
    metrics = bundle.metrics or meta.get("eval_metrics", {})
    status = str(meta.get("status") or "unknown")
    problems = meta.get("problem_names") or []
    pairs = parse_topology(meta)
    nodes = {endpoint.split(":", 1)[0] for pair in pairs for endpoint in pair}

    st.markdown('<div class="eyebrow">Network troubleshooting observatory</div>', unsafe_allow_html=True)
    st.title(_safe(meta.get("scenario_name"), "Unknown scenario"))
    st.markdown(
        f'<div class="hero-sub">Inspect the complete incident lifecycle—from topology and fault '
        f'injection to agent reasoning and evaluation.</div>'
        f'<div class="hero-meta">'
        f'<span class="chip"><span class="dot {html.escape(status)}"></span>{_safe(status).upper()}</span>'
        f'<span class="chip">Session&nbsp; {_safe(selected_id)}</span>'
        f'<span class="chip">{_safe(meta.get("scenario_topo_size"), "Fixed topology")}</span>'
        f'<span class="chip">{len(nodes)} devices · {len(pairs)} links</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    metric_cols = st.columns(6)
    headline_metrics = [
        ("Detection", _fmt_score(metrics.get("detection_score")), "Anomaly classification"),
        ("Localization F1", _fmt_score(metrics.get("localization_f1")), "Faulty device accuracy"),
        ("RCA F1", _fmt_score(metrics.get("rca_f1")), "Root-cause accuracy"),
        ("Tool calls", metrics.get("tool_calls", "—"), "MCP invocations"),
        ("Tool errors", metrics.get("tool_errors", "—"), "Failed invocations"),
        ("Steps", metrics.get("steps", "—"), "Agent reasoning steps"),
    ]
    for column, (label, value, help_text) in zip(metric_cols, headline_metrics):
        column.metric(label, value, help=help_text)

    overview_tab, topology_tab, replay_tab, trace_tab, raw_tab = st.tabs(
        ["Overview", "Topology", "Agent replay", "Agent trace", "Artifacts"]
    )

    with overview_tab:
        st.markdown('<div class="section-label">Run context</div>', unsafe_allow_html=True)
        context_left, context_right = st.columns([1.08, .92], gap="medium")
        with context_left:
            _card(
                "Experiment",
                [
                    ("Scenario", meta.get("scenario_name")),
                    ("Topology tier", meta.get("scenario_topo_size") or "Fixed"),
                    ("Lab instance", meta.get("lab_name")),
                    ("Created", _fmt_time(meta.get("created_at"))),
                    ("Problems", problems),
                ],
            )
        with context_right:
            _card(
                "Agent",
                [
                    ("Implementation", meta.get("agent_type")),
                    ("Backend", meta.get("llm_backend")),
                    ("Model", meta.get("model")),
                    ("Started", _fmt_time(meta.get("start_time"))),
                    ("Completed", _fmt_time(meta.get("end_time"))),
                ],
            )

        st.write("")
        st.markdown('<div class="section-label">Diagnosis comparison</div>', unsafe_allow_html=True)
        truth_col, prediction_col = st.columns(2, gap="medium")
        with truth_col:
            _diagnosis_card("Ground truth", bundle.ground_truth, "#ff647c")
        with prediction_col:
            _diagnosis_card("Agent submission", bundle.submission, "#ffb454")

        if bundle.ground_truth and bundle.submission:
            exact_match = (
                bundle.ground_truth.get("is_anomaly") == bundle.submission.get("is_anomaly")
                and set(bundle.ground_truth.get("faulty_devices") or [])
                == set(bundle.submission.get("faulty_devices") or [])
                and set(bundle.ground_truth.get("root_cause_name") or [])
                == set(bundle.submission.get("root_cause_name") or [])
            )
            banner_class = "match-good" if exact_match else "match-bad"
            banner_text = (
                "✓ Exact diagnosis match"
                if exact_match
                else "△ Diagnosis differs from ground truth — inspect localization and RCA metrics."
            )
            st.markdown(
                f'<div class="match-banner {banner_class}">{banner_text}</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown('<div class="section-label">Failure injections</div>', unsafe_allow_html=True)
        if bundle.failure_injections:
            compact_failures = []
            for injection in bundle.failure_injections:
                compact_failures.append(
                    {
                        "Problem": injection.get("problem_name"),
                        "Category": injection.get("root_cause_category"),
                        "Status": injection.get("status"),
                        "Started": _fmt_time(injection.get("created_at")),
                        "Parameters": json.dumps(
                            injection.get("injection_params") or {},
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )
            st.dataframe(compact_failures, hide_index=True, width="stretch")
        elif problems:
            st.info("Detailed injection records are unavailable; problem names remain in the run metadata.")
        else:
            _render_empty()

    with topology_tab:
        st.markdown(
            """
            <div class="legend-row">
              <span class="legend-chip"><span class="legend-swatch" style="background:#ff647c"></span>Ground truth</span>
              <span class="legend-chip"><span class="legend-swatch" style="background:#ffb454"></span>Agent prediction</span>
              <span class="legend-chip"><span class="legend-swatch" style="background:#a78bfa"></span>Both</span>
              <span class="legend-chip"><span style="color:#ff647c">┄</span>Fault interface</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        svg = render_topology_svg(
            pairs,
            actual_faulty=faulty_devices(bundle.ground_truth),
            predicted_faulty=faulty_devices(bundle.submission),
            fault_interfaces=fault_endpoints(bundle),
        )
        st.html(svg, width="stretch")
        st.caption("Hover nodes and links for endpoint details. Colors compare truth with agent localization.")

    with replay_tab:
        steps = replay_steps(bundle, pairs)
        if not steps:
            _render_empty()
        else:
            slider_key = f"replay_slider_{selected_id}"
            st.session_state.setdefault(slider_key, 0)
            st.session_state[slider_key] = min(
                int(st.session_state[slider_key]), len(steps) - 1
            )

            previous_col, counter_col, next_col, latest_col = st.columns(
                [0.7, 1.6, 0.7, 0.9], vertical_alignment="center"
            )
            with previous_col:
                if st.button(
                    "← Previous",
                    disabled=st.session_state[slider_key] == 0,
                    width="stretch",
                    on_click=_move_replay,
                    args=(slider_key, -1, len(steps) - 1),
                ):
                    pass
            with counter_col:
                st.slider(
                    "Replay position",
                    min_value=0,
                    max_value=len(steps) - 1,
                    format=f"Step %d / {len(steps)}",
                    label_visibility="collapsed",
                    key=slider_key,
                )
            with next_col:
                if st.button(
                    "Next →",
                    disabled=st.session_state[slider_key] >= len(steps) - 1,
                    width="stretch",
                    on_click=_move_replay,
                    args=(slider_key, 1, len(steps) - 1),
                ):
                    pass
            with latest_col:
                if st.button(
                    "Latest",
                    width="stretch",
                    on_click=_set_replay,
                    args=(slider_key, len(steps) - 1),
                ):
                    pass

            step_index = st.session_state[slider_key]
            step = steps[step_index]
            inspected = {
                device
                for previous_step in steps[: step_index + 1]
                for device in previous_step.devices
            }
            segments = "".join(
                '<span class="step-segment active"></span>'
                if index == step_index
                else '<span class="step-segment done"></span>'
                if index < step_index
                else '<span class="step-segment"></span>'
                for index in range(len(steps))
            )
            device_chips = "".join(
                f'<span class="device-chip">{html.escape(device)}</span>'
                for device in step.devices
            ) or '<span class="device-chip" style="opacity:.6">No device referenced</span>'
            st.markdown(
                f"""
                <div class="step-track">{segments}</div>
                <div class="replay-card">
                  <div class="replay-top">
                    <span class="replay-kind">{html.escape(step.kind)}</span>
                    <span class="replay-time">{html.escape(_fmt_time(step.timestamp))}</span>
                  </div>
                  <div class="replay-title">{html.escape(step.title)}</div>
                  <div class="replay-agent">{html.escape(step.agent)} · step {step_index + 1} of {len(steps)}</div>
                  <div>{device_chips}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            replay_left, replay_right = st.columns([1.25, 0.75], gap="medium")
            with replay_left:
                replay_svg = render_topology_svg(
                    pairs,
                    inspected_devices=inspected,
                    active_devices=set(step.devices),
                )
                st.html(replay_svg, width="stretch")
                st.caption(
                    "Cyan glow marks devices referenced in this step; blue marks devices inspected earlier."
                )
            with replay_right:
                if step.input:
                    st.markdown("### Input")
                    st.code(step.input, language="json")
                if step.output:
                    st.markdown("### Output")
                    st.code(step.output, language="text")
                if not step.input and not step.output:
                    st.info("This trace step contains lifecycle metadata only.")
                with st.expander("Raw event"):
                    st.json(step.raw, expanded=False)

    with trace_tab:
        rows = timeline_rows(bundle)
        agent_rows = [row for row in rows if row["source"] == "agent"]
        system_rows = [row for row in rows if row["source"] == "system"]
        tool_calls = [
            message for message in bundle.messages if message.get("event") == "tool_start"
        ]
        trace_metrics = st.columns(3)
        trace_metrics[0].metric("Timeline events", len(rows))
        trace_metrics[1].metric("Agent events", len(agent_rows))
        trace_metrics[2].metric("MCP calls", len(tool_calls))

        if rows:
            filter_col, search_col = st.columns([.65, 1.35])
            with filter_col:
                source_filter = st.multiselect(
                    "Source", ["system", "agent"], default=["system", "agent"]
                )
            with search_col:
                search = st.text_input("Search timeline", placeholder="ping_pair, failure, diagnosis…")
            filtered = [
                row
                for row in rows
                if row["source"] in source_filter
                and (
                    not search
                    or search.lower()
                    in " ".join(str(value) for value in row.values()).lower()
                )
            ]
            st.dataframe(
                filtered,
                hide_index=True,
                width="stretch",
                height=470,
                column_order=["timestamp", "source", "actor", "event", "detail"],
                column_config={
                    "timestamp": st.column_config.TextColumn("Time", width="medium"),
                    "source": st.column_config.TextColumn("Source", width="small"),
                    "actor": st.column_config.TextColumn("Actor", width="medium"),
                    "event": st.column_config.TextColumn("Event", width="medium"),
                    "detail": st.column_config.TextColumn("Details", width="large"),
                },
            )
        else:
            _render_empty()

        if tool_calls:
            st.write("")
            st.markdown(
                f'<div class="section-label">MCP tool calls · {len(tool_calls)}</div>',
                unsafe_allow_html=True,
            )
            for index, call in enumerate(tool_calls, start=1):
                tool = call.get("tool") or {}
                tool_name = tool.get("name", "unknown") if isinstance(tool, dict) else str(tool)
                agent_name = call.get("agent", "agent")
                with st.expander(f"{index:02d}  ·  {agent_name}  →  {tool_name}"):
                    st.code(str(call.get("input") or "{}"), language="json")

    with raw_tab:
        artifacts = {
            "run.json": meta,
            "ground_truth.json": bundle.ground_truth,
            "submission.json": bundle.submission,
            "eval_metrics.json": bundle.metrics,
            "llm_judge.json": bundle.judge,
            "events.jsonl": bundle.events,
            "messages.jsonl": bundle.messages,
        }
        artifact_col, download_col = st.columns([1.4, .6], vertical_alignment="bottom")
        with artifact_col:
            selected_artifact = st.selectbox("Artifact", list(artifacts))
        with download_col:
            st.download_button(
                "↓  Download snapshot",
                data=json.dumps(artifacts, indent=2, ensure_ascii=False, default=str),
                file_name=f"{selected_id}-snapshot.json",
                mime="application/json",
                width="stretch",
            )
        st.json(artifacts[selected_artifact], expanded=False)


_render_dashboard()
