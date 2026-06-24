"""Render a compact, dependency-free SVG network topology."""

from __future__ import annotations

import html
import math
import re

from nika.visualization.data import endpoint_parts


def _node_kind(name: str) -> str:
    lowered = name.lower()
    if "controller" in lowered:
        return "controller"
    if any(token in lowered for token in ("server", "dns", "dhcp", "web", "vpn", "lb", "influx")):
        return "server"
    if "router" in lowered or re.match(r"^r\d", lowered) or lowered.startswith(
        ("spine", "leaf", "super_spine")
    ):
        return "router"
    if "switch" in lowered or re.match(r"^(sw|s)\d", lowered):
        return "switch"
    return "host"


def _force_layout(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    """Small deterministic force layout suitable for dashboard-sized graphs."""
    count = len(nodes)
    if count == 1:
        return {nodes[0]: (0.5, 0.5)}

    positions = {
        node: (
            0.5 + 0.38 * math.cos(2 * math.pi * index / count),
            0.5 + 0.38 * math.sin(2 * math.pi * index / count),
        )
        for index, node in enumerate(nodes)
    }
    area = 1.0
    ideal = math.sqrt(area / max(count, 1))

    for iteration in range(90):
        displacement = {node: [0.0, 0.0] for node in nodes}
        for index, left in enumerate(nodes):
            for right in nodes[index + 1 :]:
                dx = positions[left][0] - positions[right][0]
                dy = positions[left][1] - positions[right][1]
                distance = max(math.hypot(dx, dy), 0.01)
                force = ideal * ideal / distance
                fx, fy = dx / distance * force, dy / distance * force
                displacement[left][0] += fx
                displacement[left][1] += fy
                displacement[right][0] -= fx
                displacement[right][1] -= fy

        for left, right in edges:
            dx = positions[left][0] - positions[right][0]
            dy = positions[left][1] - positions[right][1]
            distance = max(math.hypot(dx, dy), 0.01)
            force = distance * distance / ideal
            fx, fy = dx / distance * force, dy / distance * force
            displacement[left][0] -= fx
            displacement[left][1] -= fy
            displacement[right][0] += fx
            displacement[right][1] += fy

        temperature = 0.08 * (1 - iteration / 90)
        for node in nodes:
            dx, dy = displacement[node]
            magnitude = max(math.hypot(dx, dy), 0.01)
            x = positions[node][0] + dx / magnitude * min(magnitude, temperature)
            y = positions[node][1] + dy / magnitude * min(magnitude, temperature)
            positions[node] = (min(0.93, max(0.07, x)), min(0.90, max(0.10, y)))

    return positions


def render_topology_svg(
    endpoint_pairs: list[tuple[str, str]],
    *,
    actual_faulty: set[str] | None = None,
    predicted_faulty: set[str] | None = None,
    fault_interfaces: set[tuple[str, str]] | None = None,
    inspected_devices: set[str] | None = None,
    active_devices: set[str] | None = None,
) -> str:
    """Return an embeddable HTML document containing an SVG graph."""
    actual_faulty = actual_faulty or set()
    predicted_faulty = predicted_faulty or set()
    fault_interfaces = fault_interfaces or set()
    inspected_devices = inspected_devices or set()
    active_devices = active_devices or set()

    parsed_edges: list[tuple[str, str, str, str]] = []
    node_names: set[str] = set()
    for left_endpoint, right_endpoint in endpoint_pairs:
        left, left_intf = endpoint_parts(left_endpoint)
        right, right_intf = endpoint_parts(right_endpoint)
        if not left or not right:
            continue
        parsed_edges.append((left, right, left_intf, right_intf))
        node_names.update((left, right))

    if not node_names:
        return """
        <div style="height:500px;display:grid;place-items:center;border:1px dashed rgba(142,164,190,.28);
                    border-radius:18px;color:#8d9bb0;font-family:Inter,ui-sans-serif,system-ui;
                    background:linear-gradient(145deg,rgba(16,29,47,.75),rgba(7,16,29,.9))">
          <div style="text-align:center"><div style="font-size:2rem;margin-bottom:.6rem">◇</div>
          <b style="color:#dce7f5">Topology is not available yet</b><br>
          <span style="font-size:.85rem">New sessions save a topology snapshot automatically.</span></div>
        </div>
        """

    nodes = sorted(node_names)
    simple_edges = [(left, right) for left, right, _, _ in parsed_edges]
    positions = _force_layout(nodes, simple_edges)
    width, height = 1080, 590
    margin_x, margin_y = 70, 58

    def xy(node: str) -> tuple[float, float]:
        x, y = positions[node]
        return margin_x + x * (width - 2 * margin_x), margin_y + y * (height - 2 * margin_y)

    edge_markup: list[str] = []
    for left, right, left_intf, right_intf in parsed_edges:
        x1, y1 = xy(left)
        x2, y2 = xy(right)
        failed = (left, left_intf) in fault_interfaces or (right, right_intf) in fault_interfaces
        color = "#ff647c" if failed else "#4c617a"
        dash = ' stroke-dasharray="10 7"' if failed else ""
        glow = ' filter="url(#faultGlow)"' if failed else ""
        title = html.escape(f"{left}:{left_intf} ↔ {right}:{right_intf}")
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        interface_label = html.escape(f"{left_intf} · {right_intf}")
        edge_markup.append(
            f'<g><title>{title}</title><line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.5"{dash}{glow}/>'
            f'<rect x="{mid_x - 31:.1f}" y="{mid_y - 10:.1f}" width="62" height="19" rx="7" '
            f'fill="#0b1728" stroke="#263a52" stroke-width=".7"/>'
            f'<text x="{mid_x:.1f}" y="{mid_y + 3.5:.1f}" text-anchor="middle" '
            f'fill="#8091a8" font-size="9.5">{interface_label}</text></g>'
        )

    palettes = {
        "host": ("#123d43", "#55e1d4", "H"),
        "router": ("#173a68", "#72adff", "R"),
        "switch": ("#34255e", "#b49afa", "S"),
        "server": ("#16425d", "#67c7f1", "D"),
        "controller": ("#52213e", "#f48ab9", "C"),
    }
    node_markup: list[str] = []
    for node in nodes:
        x, y = xy(node)
        fill, stroke, symbol = palettes[_node_kind(node)]
        if node in actual_faulty and node in predicted_faulty:
            fill, stroke = "#5b38a6", "#c4b5fd"
        elif node in actual_faulty:
            fill, stroke = "#70263a", "#ff7c91"
        elif node in predicted_faulty:
            fill, stroke = "#704817", "#ffc36a"
        elif node in active_devices:
            fill, stroke = "#155761", "#62f2e5"
        elif node in inspected_devices:
            fill, stroke = "#173b52", "#59b7da"
        safe_name = html.escape(node)
        active_glow = ' filter="url(#activeGlow)"' if node in active_devices else ""
        node_markup.append(
            f'<g><title>{safe_name}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="31" fill="#06101d" opacity=".55"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="27" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2.5"{active_glow}/>'
            f'<circle cx="{x - 18:.1f}" cy="{y - 18:.1f}" r="4" fill="{stroke}"/>'
            f'<text x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle" fill="#f3f7fc" '
            f'font-size="15" font-weight="800">{symbol}</text>'
            f'<rect x="{x - 44:.1f}" y="{y + 37:.1f}" width="88" height="25" rx="9" '
            f'fill="#0b1728" stroke="#263a52" stroke-width=".8"/>'
            f'<text x="{x:.1f}" y="{y + 54:.1f}" text-anchor="middle" fill="#dce7f5" '
            f'font-size="12" font-weight="650">{safe_name}</text></g>'
        )

    return f"""
    <div style="background:linear-gradient(145deg,#0d1b2d,#07111f);border:1px solid #20334b;
                border-radius:18px;overflow:hidden;box-shadow:0 20px 50px rgba(0,0,0,.18)">
      <svg viewBox="0 0 {width} {height}" width="100%" height="590"
           xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Network topology">
        <defs>
          <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#20334b" stroke-width=".55" opacity=".42"/>
          </pattern>
          <filter id="faultGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="activeGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <radialGradient id="surfaceGlow" cx="50%" cy="45%" r="65%">
            <stop offset="0%" stop-color="#142a42"/><stop offset="100%" stop-color="#081321"/>
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#surfaceGlow)"/>
        <rect width="100%" height="100%" fill="url(#grid)"/>
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """
