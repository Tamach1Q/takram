#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz


ROUTE_CLASSES = {"route_candidate_dashed", "route_candidate_solid_aux"}
TARGET_DASHES = {"[ .05 2.5 ] 0", "[ .02 3 ] 0"}
MAPBOX_SCALE = 0.01


@dataclass
class RouteSegment:
    segment_id: str
    page_no: int
    frame_id: str | None
    draw_index: int
    polyline_index: int
    style_class: str
    classification: str
    width_pt: float
    dashes: str
    points_pdf: list[tuple[float, float]]
    length_pt: float
    confidence: float
    needs_manual_review: bool
    review_reasons: list[str] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.points_pdf]
        ys = [point[1] for point in self.points_pdf]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class MergedRoute:
    route_id: str
    page_no: int
    frame_id: str | None
    frame_source_kind: str | None
    style_class: str
    chains_pdf: list[list[tuple[float, float]]]
    segment_ids: list[str]
    segment_count: int
    branch_node_count: int
    chain_count: int
    confidence: float
    needs_manual_review: bool
    review_reasons: list[str]
    component_bbox: tuple[float, float, float, float]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def point_tuple(point: Any) -> tuple[float, float]:
    return (float(point.x), float(point.y))


def polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.dist(start, end)
    return total


def normalize_dash(value: Any) -> str:
    if value in (None, "", "[] 0"):
        return "solid"
    return str(value).strip()


def sample_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 16,
) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for step in range(steps + 1):
        t = step / steps
        mt = 1.0 - t
        x = (
            (mt ** 3) * p0[0]
            + 3 * (mt ** 2) * t * p1[0]
            + 3 * mt * (t ** 2) * p2[0]
            + (t ** 3) * p3[0]
        )
        y = (
            (mt ** 3) * p0[1]
            + 3 * (mt ** 2) * t * p1[1]
            + 3 * mt * (t ** 2) * p2[1]
            + (t ** 3) * p3[1]
        )
        sampled.append((x, y))
    return sampled


def polylines_from_items(items: list[Any]) -> list[list[tuple[float, float]]]:
    polylines: list[list[tuple[float, float]]] = []
    current_path_start: tuple[float, float] | None = None
    current_point: tuple[float, float] | None = None
    current_polyline: list[tuple[float, float]] = []

    def same_point(left: tuple[float, float], right: tuple[float, float], tolerance: float = 0.01) -> bool:
        return math.dist(left, right) <= tolerance

    def flush_current() -> None:
        nonlocal current_polyline
        if len(current_polyline) >= 2:
            polylines.append(current_polyline)
        current_polyline = []

    for item in items:
        operator = item[0]
        if operator == "l":
            start = point_tuple(item[1])
            end = point_tuple(item[2])
            if not current_polyline:
                current_polyline = [start, end]
            elif same_point(current_polyline[-1], start):
                current_polyline.append(end)
            else:
                flush_current()
                current_polyline = [start, end]
            if current_path_start is None:
                current_path_start = start
            current_point = end
        elif operator == "c":
            p0 = point_tuple(item[1])
            p1 = point_tuple(item[2])
            p2 = point_tuple(item[3])
            p3 = point_tuple(item[4])
            sampled = sample_cubic_bezier(p0, p1, p2, p3)
            if not current_polyline:
                current_polyline = sampled
            elif same_point(current_polyline[-1], sampled[0]):
                current_polyline.extend(sampled[1:])
            else:
                flush_current()
                current_polyline = sampled
            if current_path_start is None:
                current_path_start = p0
            current_point = p3
        elif operator == "re":
            flush_current()
            rect = item[1]
            polylines.append(
                [
                    (float(rect.x0), float(rect.y0)),
                    (float(rect.x1), float(rect.y0)),
                    (float(rect.x1), float(rect.y1)),
                    (float(rect.x0), float(rect.y1)),
                    (float(rect.x0), float(rect.y0)),
                ]
            )
            current_path_start = None
            current_point = None
        elif operator == "qu":
            flush_current()
            quad = item[1]
            polylines.append(
                [
                    point_tuple(quad.ul),
                    point_tuple(quad.ur),
                    point_tuple(quad.lr),
                    point_tuple(quad.ll),
                    point_tuple(quad.ul),
                ]
            )
            current_path_start = None
            current_point = None
        elif operator == "h":
            if current_path_start is not None and current_point is not None and not same_point(current_point, current_path_start):
                if not current_polyline:
                    current_polyline = [current_point, current_path_start]
                else:
                    current_polyline.append(current_path_start)
            flush_current()
            current_path_start = None
            current_point = None
    flush_current()
    return polylines


def to_synthetic_coords(points_pdf: list[tuple[float, float]]) -> list[list[float]]:
    return [[round(x * MAPBOX_SCALE, 6), round(-y * MAPBOX_SCALE, 6)] for x, y in points_pdf]


def bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_route_rows(step2_dir: Path) -> list[dict[str, Any]]:
    rows = load_csv_rows(step2_dir / "red_objects_with_frames.csv")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if row["classification"] not in ROUTE_CLASSES:
            continue
        for field in [
            "page_no",
            "draw_index",
            "width_pt",
            "bbox_x0_pt",
            "bbox_y0_pt",
            "bbox_x1_pt",
            "bbox_y1_pt",
            "bbox_w_pt",
            "bbox_h_pt",
            "path_length_pt",
        ]:
            if field in {"page_no", "draw_index"}:
                row[field] = int(float(row[field]))
            else:
                row[field] = float(row[field])
        parsed.append(row)
    return parsed


def load_frames(path: Path) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for row in load_csv_rows(path):
        frames[row["frame_id"]] = row
    return frames


def style_class_from_row(row: dict[str, Any]) -> str:
    if row["classification"] == "route_candidate_dashed":
        return "walk_main"
    if row["classification"] == "route_candidate_solid_aux":
        return "walk_sub"
    return "unknown_red"


def segment_confidence(row: dict[str, Any], length_pt: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    confidence = 0.45
    if row["classification"] == "route_candidate_dashed":
        confidence += 0.28
        if row["dashes"] in TARGET_DASHES:
            confidence += 0.12
        if abs(row["width_pt"] - 1.25) <= 0.08 or abs(row["width_pt"] - 2.0) <= 0.08:
            confidence += 0.08
    elif row["classification"] == "route_candidate_solid_aux":
        confidence += 0.12
        if abs(row["width_pt"] - 0.85) <= 0.08:
            confidence += 0.10
        if length_pt < 18:
            reasons.append("short_aux_segment")
            confidence -= 0.10
    if length_pt >= 40:
        confidence += 0.06
    elif length_pt < 10:
        reasons.append("very_short_segment")
        confidence -= 0.16
    if not row.get("frame_id"):
        reasons.append("missing_frame_id")
        confidence -= 0.18
    return max(0.0, min(confidence, 0.98)), reasons


def endpoint_vector(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        return (0.0, 0.0)
    return (points[1][0] - points[0][0], points[1][1] - points[0][1])


def vector_angle_score(a: tuple[float, float], b: tuple[float, float]) -> float:
    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)


def build_nodes_for_segments(
    segments: list[RouteSegment],
    snap_threshold: float,
) -> tuple[dict[str, tuple[int, int]], dict[int, tuple[float, float]], dict[int, list[str]]]:
    node_centers: dict[int, tuple[float, float]] = {}
    node_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    segment_nodes: dict[str, tuple[int, int]] = {}
    next_node_id = 1

    def assign_node(point: tuple[float, float]) -> int:
        nonlocal next_node_id
        for node_id, center in node_centers.items():
            if math.dist(point, center) <= snap_threshold:
                node_points[node_id].append(point)
                avg_x = sum(item[0] for item in node_points[node_id]) / len(node_points[node_id])
                avg_y = sum(item[1] for item in node_points[node_id]) / len(node_points[node_id])
                node_centers[node_id] = (avg_x, avg_y)
                return node_id
        node_id = next_node_id
        next_node_id += 1
        node_centers[node_id] = point
        node_points[node_id].append(point)
        return node_id

    adjacency: dict[int, list[str]] = defaultdict(list)
    for segment in segments:
        start_node = assign_node(segment.points_pdf[0])
        end_node = assign_node(segment.points_pdf[-1])
        segment_nodes[segment.segment_id] = (start_node, end_node)
        adjacency[start_node].append(segment.segment_id)
        adjacency[end_node].append(segment.segment_id)
    return segment_nodes, node_centers, adjacency


def connected_components(
    segment_ids: list[str],
    adjacency: dict[int, list[str]],
    segment_nodes: dict[str, tuple[int, int]],
) -> list[list[str]]:
    visited: set[str] = set()
    components: list[list[str]] = []
    node_to_segments = adjacency

    for segment_id in segment_ids:
        if segment_id in visited:
            continue
        stack = [segment_id]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            start_node, end_node = segment_nodes[current]
            for node_id in (start_node, end_node):
                for neighbor in node_to_segments[node_id]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        components.append(component)
    return components


def other_node(segment_nodes: dict[str, tuple[int, int]], segment_id: str, node_id: int) -> int:
    start_node, end_node = segment_nodes[segment_id]
    return end_node if start_node == node_id else start_node


def oriented_points(segment: RouteSegment, segment_nodes: dict[str, tuple[int, int]], from_node: int) -> list[tuple[float, float]]:
    start_node, _ = segment_nodes[segment.segment_id]
    if start_node == from_node:
        return list(segment.points_pdf)
    return list(reversed(segment.points_pdf))


def build_chains(
    component_ids: list[str],
    segments_by_id: dict[str, RouteSegment],
    adjacency: dict[int, list[str]],
    segment_nodes: dict[str, tuple[int, int]],
) -> list[list[tuple[float, float]]]:
    degrees = {node_id: len([segment_id for segment_id in incident if segment_id in component_ids]) for node_id, incident in adjacency.items()}
    unused = set(component_ids)
    chains: list[list[tuple[float, float]]] = []

    def choose_next(current_node: int, prev_vec: tuple[float, float] | None) -> str | None:
        candidates = [segment_id for segment_id in adjacency[current_node] if segment_id in unused]
        if not candidates:
            return None
        if prev_vec is None:
            candidates.sort(key=lambda segment_id: segments_by_id[segment_id].length_pt, reverse=True)
            return candidates[0]
        scored: list[tuple[float, float, str]] = []
        for segment_id in candidates:
            points = oriented_points(segments_by_id[segment_id], segment_nodes, current_node)
            next_vec = endpoint_vector(points)
            scored.append((vector_angle_score(prev_vec, next_vec), segments_by_id[segment_id].length_pt, segment_id))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    while unused:
        available_nodes = sorted(
            [
                node_id
                for node_id, degree in degrees.items()
                if degree != 2 and any(segment_id in unused for segment_id in adjacency[node_id])
            ]
        )
        if available_nodes:
            start_node = available_nodes[0]
        else:
            any_segment = segments_by_id[next(iter(unused))]
            start_node = segment_nodes[any_segment.segment_id][0]

        chain: list[tuple[float, float]] = []
        current_node = start_node
        prev_vec: tuple[float, float] | None = None
        traversed_any = False

        while True:
            next_segment_id = choose_next(current_node, prev_vec)
            if next_segment_id is None:
                break
            segment = segments_by_id[next_segment_id]
            points = oriented_points(segment, segment_nodes, current_node)
            if not chain:
                chain.extend(points)
            else:
                chain.extend(points[1:])
            unused.remove(next_segment_id)
            traversed_any = True
            prev_vec = (chain[-1][0] - chain[-2][0], chain[-1][1] - chain[-2][1]) if len(chain) >= 2 else None
            current_node = other_node(segment_nodes, next_segment_id, current_node)
            if degrees.get(current_node, 0) != 2 and traversed_any:
                break
        if chain:
            chains.append(chain)
        else:
            break
    return chains


def component_confidence(segments: list[RouteSegment], branch_node_count: int, chain_count: int, frame_source_kind: str | None) -> tuple[float, list[str]]:
    reasons: list[str] = []
    confidence = sum(segment.confidence for segment in segments) / max(len(segments), 1)
    if branch_node_count > 0:
        reasons.append("branching_component")
        confidence -= 0.10
    if chain_count > 1:
        reasons.append("multi_chain_component")
        confidence -= 0.07
    if frame_source_kind == "cluster_fallback":
        reasons.append("fallback_frame")
        confidence -= 0.08
    if any(segment.needs_manual_review for segment in segments):
        reasons.append("contains_low_confidence_segment")
        confidence -= 0.06
    return max(0.0, min(confidence, 0.98)), reasons


def page_image_manifest_entry(page_no: int, page_rect: fitz.Rect, image_name: str) -> dict[str, Any]:
    lon_max = round(float(page_rect.width) * MAPBOX_SCALE, 6)
    lat_max = round(float(page_rect.height) * MAPBOX_SCALE, 6)
    return {
        "page_no": page_no,
        "width_pt": round(float(page_rect.width), 3),
        "height_pt": round(float(page_rect.height), 3),
        "image_path": f"pages/{image_name}",
        "image_coordinates": [[0.0, 0.0], [lon_max, 0.0], [lon_max, -lat_max], [0.0, -lat_max]],
        "fit_bounds": [[0.0, -lat_max], [lon_max, 0.0]],
    }


def load_env_token(env_path: Path) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "MAPBOX_ACCESS_TOKEN":
            return value.strip()
    return None


def make_geojson_feature(geometry: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manual_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["issue_id", "object_type", "object_id", "page_no", "frame_id", "severity", "reason", "details"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_html(debug_dir: Path, config: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Step 3 Mapbox Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    :root {{
      --bg: #f0ede5;
      --panel: #fffaf0;
      --ink: #1f1f1a;
      --accent: #b23a48;
      --line: #d9d2bf;
    }}
    html, body {{
      margin: 0;
      height: 100%;
      background: var(--bg);
      color: var(--ink);
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-columns: 320px 1fr;
      height: 100%;
    }}
    #sidebar {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 14px;
    }}
    label {{
      display: block;
      font-size: 12px;
      margin: 12px 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select, button {{
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
      font-size: 14px;
    }}
    .toggle {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }}
    .toggle button {{
      width: auto;
      flex: 1;
    }}
    .toggle button.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .metric {{
      margin-top: 14px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
      margin-bottom: 4px;
    }}
    ul {{
      padding-left: 18px;
    }}
    .warning {{
      color: #a12b2b;
      font-weight: 700;
    }}
    .code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      background: #fcfaf5;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h1>Step 3 Mapbox Debug</h1>
      <label for="pageSelect">Page</label>
      <select id="pageSelect"></select>
      <label for="frameSelect">Frame</label>
      <select id="frameSelect"></select>
      <div class="toggle">
        <button id="mergedBtn" class="active">Merged</button>
        <button id="rawBtn">Raw</button>
      </div>
      <div class="metric"><strong id="segmentCount">0</strong>segment count</div>
      <div class="metric"><strong id="routeCount">0</strong>merged route count</div>
      <div class="metric"><strong id="issueCount">0</strong>manual review issues</div>
      <div id="summary" class="code"></div>
      <div id="issues" class="code"></div>
    </aside>
    <main id="map"></main>
  </div>
  <script>window.DEBUG_CONFIG = {json.dumps(config, ensure_ascii=False)};</script>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js"></script>
  <script>
    const blankStyle = {{
      version: 8,
      sources: {{}},
      layers: [{{ id: 'background', type: 'background', paint: {{ 'background-color': '#f6f2ea' }} }}]
    }};

    const mapboxToken = window.DEBUG_CONFIG.mapboxToken || '';
    if (mapboxToken) {{
      mapboxgl.accessToken = mapboxToken;
    }}

    const state = {{
      showRaw: false,
      pages: [],
      raw: [],
      merged: [],
      issues: [],
      currentPage: null,
      currentFrame: 'all',
      imageLoadedFor: null,
      logs: []
    }};

    const els = {{
      pageSelect: document.getElementById('pageSelect'),
      frameSelect: document.getElementById('frameSelect'),
      mergedBtn: document.getElementById('mergedBtn'),
      rawBtn: document.getElementById('rawBtn'),
      segmentCount: document.getElementById('segmentCount'),
      routeCount: document.getElementById('routeCount'),
      issueCount: document.getElementById('issueCount'),
      summary: document.getElementById('summary'),
      issues: document.getElementById('issues'),
    }};

    function logIssue(message) {{
      state.logs.push(message);
      renderDiagnostics();
    }}

    const map = new mapboxgl.Map({{
      container: 'map',
      style: blankStyle,
      center: [5, -3],
      zoom: 4,
      attributionControl: false
    }});

    map.on('error', event => {{
      if (event && event.error) {{
        logIssue('map error: ' + event.error.message);
      }}
    }});

    async function loadData() {{
      const [pagesRes, rawRes, mergedRes, issuesRes] = await Promise.all([
        fetch('page_manifest.json'),
        fetch('raw_segments.geojson'),
        fetch('merged_routes.geojson'),
        fetch('manual_review.json')
      ]);
      state.pages = await pagesRes.json();
      state.raw = (await rawRes.json()).features;
      state.merged = (await mergedRes.json()).features;
      state.issues = await issuesRes.json();
      initSelectors();
      state.currentPage = state.pages[0]?.page_no || null;
      updateFrameOptions();
      updatePage();
    }}

    function initSelectors() {{
      els.pageSelect.innerHTML = '';
      for (const page of state.pages) {{
        const opt = document.createElement('option');
        opt.value = String(page.page_no);
        opt.textContent = `Page ${{page.page_no}}`;
        els.pageSelect.appendChild(opt);
      }}
      els.pageSelect.addEventListener('change', () => {{
        state.currentPage = Number(els.pageSelect.value);
        state.currentFrame = 'all';
        updateFrameOptions();
        updatePage();
      }});
      els.frameSelect.addEventListener('change', () => {{
        state.currentFrame = els.frameSelect.value;
        updateLayers();
        renderDiagnostics();
      }});
      els.mergedBtn.addEventListener('click', () => {{
        state.showRaw = false;
        els.mergedBtn.classList.add('active');
        els.rawBtn.classList.remove('active');
        updateLayers();
      }});
      els.rawBtn.addEventListener('click', () => {{
        state.showRaw = true;
        els.rawBtn.classList.add('active');
        els.mergedBtn.classList.remove('active');
        updateLayers();
      }});
    }}

    function updateFrameOptions() {{
      const merged = state.merged.filter(f => f.properties.page_no === state.currentPage);
      const frameIds = Array.from(new Set(merged.map(f => f.properties.frame_id).filter(Boolean))).sort();
      els.frameSelect.innerHTML = '';
      const all = document.createElement('option');
      all.value = 'all';
      all.textContent = 'All frames';
      els.frameSelect.appendChild(all);
      for (const frameId of frameIds) {{
        const opt = document.createElement('option');
        opt.value = frameId;
        opt.textContent = frameId;
        els.frameSelect.appendChild(opt);
      }}
      els.frameSelect.value = 'all';
    }}

    function currentPageMeta() {{
      return state.pages.find(page => page.page_no === state.currentPage);
    }}

    function filteredFeatures(source) {{
      return source.filter(feature => {{
        if (feature.properties.page_no !== state.currentPage) return false;
        if (state.currentFrame === 'all') return true;
        return feature.properties.frame_id === state.currentFrame;
      }});
    }}

    function updateImageSource() {{
      const meta = currentPageMeta();
      if (!meta) return;
      const sourceId = 'page-image';
      const source = map.getSource(sourceId);
      const imageUrl = meta.image_path;
      if (source) {{
        source.updateImage({{ url: imageUrl, coordinates: meta.image_coordinates }});
      }} else {{
        map.addSource(sourceId, {{ type: 'image', url: imageUrl, coordinates: meta.image_coordinates }});
        map.addLayer({{ id: 'page-image-layer', type: 'raster', source: sourceId, paint: {{ 'raster-opacity': 0.92 }} }});
      }}
      map.fitBounds(meta.fit_bounds, {{ padding: 40, duration: 0 }});
    }}

    function updateSource(sourceId, features) {{
      const collection = {{ type: 'FeatureCollection', features }};
      if (map.getSource(sourceId)) {{
        map.getSource(sourceId).setData(collection);
      }} else {{
        map.addSource(sourceId, {{ type: 'geojson', data: collection }});
      }}
    }}

    function ensureLayers() {{
      if (!map.getLayer('merged-main')) {{
        map.addLayer({{
          id: 'merged-main',
          type: 'line',
          source: 'merged-routes',
          paint: {{
            'line-color': ['case', ['boolean', ['get', 'needs_manual_review'], false], '#c73e1d', '#0f7b6c'],
            'line-width': ['case', ['==', ['get', 'style_class'], 'walk_main'], 4, 3],
            'line-dasharray': ['case', ['boolean', ['get', 'needs_manual_review'], false], ['literal', [0.8, 1.2]], ['literal', [1, 0]]]
          }}
        }});
      }}
      if (!map.getLayer('raw-main')) {{
        map.addLayer({{
          id: 'raw-main',
          type: 'line',
          source: 'raw-segments',
          layout: {{ visibility: 'none' }},
          paint: {{
            'line-color': ['case', ['==', ['get', 'style_class'], 'walk_main'], '#1f78b4', '#7fc97f'],
            'line-width': 2,
            'line-dasharray': ['case', ['==', ['get', 'style_class'], 'walk_main'], ['literal', [1.2, 1.0]], ['literal', [1, 0]]]
          }}
        }});
      }}
    }}

    function updateLayers() {{
      const raw = filteredFeatures(state.raw);
      const merged = filteredFeatures(state.merged);
      updateSource('raw-segments', raw);
      updateSource('merged-routes', merged);
      ensureLayers();
      map.setLayoutProperty('raw-main', 'visibility', state.showRaw ? 'visible' : 'none');
      map.setLayoutProperty('merged-main', 'visibility', state.showRaw ? 'none' : 'visible');
      els.segmentCount.textContent = String(raw.length);
      els.routeCount.textContent = String(merged.length);
      renderDiagnostics();
    }}

    function renderDiagnostics() {{
      const raw = filteredFeatures(state.raw);
      const merged = filteredFeatures(state.merged);
      const issues = state.issues.filter(issue => issue.page_no === state.currentPage && (state.currentFrame === 'all' || issue.frame_id === state.currentFrame));
      els.issueCount.textContent = String(issues.length);
      const summary = {{
        page: state.currentPage,
        frame: state.currentFrame,
        raw_segments: raw.length,
        merged_routes: merged.length,
        manual_review: issues.length,
        log: state.logs.slice(-5)
      }};
      els.summary.textContent = JSON.stringify(summary, null, 2);
      if (issues.length === 0) {{
        els.issues.textContent = 'manual review issue: none';
      }} else {{
        els.issues.textContent = issues.map(issue => `${{issue.object_type}} ${{issue.object_id}}: ${{issue.reason}}${{issue.details ? ' / ' + issue.details : ''}}`).join('\\n');
      }}
    }}

    function attachPopups() {{
      const layers = ['merged-main', 'raw-main'];
      for (const layerId of layers) {{
        map.on('click', layerId, event => {{
          const feature = event.features?.[0];
          if (!feature) return;
          const props = feature.properties || {{}};
          const html = `<div style="font-size:12px;line-height:1.5"><strong>${{props.route_id || props.segment_id || 'item'}}</strong><br/>page=${{props.page_no}}<br/>frame=${{props.frame_id || 'none'}}<br/>style=${{props.style_class}}<br/>review=${{props.needs_manual_review}}</div>`;
          new mapboxgl.Popup().setLngLat(event.lngLat).setHTML(html).addTo(map);
        }});
      }}
    }}

    function updatePage() {{
      els.pageSelect.value = String(state.currentPage);
      updateImageSource();
      updateLayers();
    }}

    map.on('load', async () => {{
      try {{
        await loadData();
        attachPopups();
      }} catch (error) {{
        logIssue('data load failed: ' + error.message);
      }}
    }});
  </script>
</body>
</html>
"""
    (debug_dir / "index.html").write_text(html, encoding="utf-8")


def build_step3(pdf_path: Path, step2_dir: Path, out_dir: Path, env_path: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    preview_dir = out_dir / "pages"
    ensure_dir(preview_dir)
    debug_dir = out_dir / "mapbox_debug"
    ensure_dir(debug_dir)
    ensure_dir(debug_dir / "pages")

    route_rows = load_route_rows(step2_dir)
    frames = load_frames(step2_dir / "frames.csv")
    frame_source_by_id = {frame_id: row.get("source_kind") for frame_id, row in frames.items()}

    doc = fitz.open(pdf_path)
    page_drawings: dict[int, list[Any]] = {}
    page_meta: dict[int, dict[str, Any]] = {}
    raw_segments: list[RouteSegment] = []
    manual_review_rows: list[dict[str, Any]] = []
    issue_index = 1

    def add_issue(
        object_type: str,
        object_id: str,
        page_no: int,
        frame_id: str | None,
        severity: str,
        reason: str,
        details: str,
    ) -> None:
        nonlocal issue_index
        manual_review_rows.append(
            {
                "issue_id": f"issue_{issue_index:05d}",
                "object_type": object_type,
                "object_id": object_id,
                "page_no": page_no,
                "frame_id": frame_id or "",
                "severity": severity,
                "reason": reason,
                "details": details,
            }
        )
        issue_index += 1

    segment_counter = 1
    for row in route_rows:
        page_no = row["page_no"]
        if page_no not in page_drawings:
            page = doc.load_page(page_no - 1)
            page_drawings[page_no] = page.get_drawings()
            page_meta[page_no] = {"width_pt": float(page.rect.width), "height_pt": float(page.rect.height)}
        drawing = page_drawings[page_no][row["draw_index"]]
        polylines = polylines_from_items(drawing.get("items", []))
        if not polylines:
            continue
        for polyline_index, points in enumerate(polylines):
            if len(points) < 2:
                continue
            length_pt = polyline_length(points)
            if length_pt < 4.0:
                continue
            confidence, confidence_reasons = segment_confidence(row, length_pt)
            style_class = style_class_from_row(row)
            review_reasons = list(confidence_reasons)
            if confidence < 0.58:
                review_reasons.append("low_confidence")
            segment = RouteSegment(
                segment_id=f"seg_{segment_counter:06d}",
                page_no=page_no,
                frame_id=row.get("frame_id") or None,
                draw_index=row["draw_index"],
                polyline_index=polyline_index,
                style_class=style_class,
                classification=row["classification"],
                width_pt=row["width_pt"],
                dashes=row["dashes"],
                points_pdf=points,
                length_pt=length_pt,
                confidence=confidence,
                needs_manual_review=bool(review_reasons),
                review_reasons=review_reasons,
            )
            raw_segments.append(segment)
            segment_counter += 1
            for reason in review_reasons:
                add_issue(
                    object_type="segment",
                    object_id=segment.segment_id,
                    page_no=segment.page_no,
                    frame_id=segment.frame_id,
                    severity="warning" if reason != "missing_frame_id" else "error",
                    reason=reason,
                    details=f"length_pt={segment.length_pt:.2f}, width_pt={segment.width_pt:.2f}, dash={segment.dashes}",
                )

    raw_by_group: dict[tuple[int, str | None], list[RouteSegment]] = defaultdict(list)
    for segment in raw_segments:
        raw_by_group[(segment.page_no, segment.frame_id)].append(segment)

    merged_routes: list[MergedRoute] = []
    route_counter_by_page: Counter[int] = Counter()
    for (page_no, frame_id), segments in sorted(raw_by_group.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        if not frame_id:
            continue
        snap_threshold = 6.5 if any(segment.style_class == "walk_main" for segment in segments) else 8.0
        segment_nodes, _, adjacency = build_nodes_for_segments(segments, snap_threshold=snap_threshold)
        component_ids = connected_components([segment.segment_id for segment in segments], adjacency, segment_nodes)
        segments_by_id = {segment.segment_id: segment for segment in segments}
        for component in component_ids:
            component_segments = [segments_by_id[segment_id] for segment_id in component]
            chain_points = build_chains(component, segments_by_id, adjacency, segment_nodes)
            component_nodes = set()
            for segment_id in component:
                component_nodes.update(segment_nodes[segment_id])
            branch_node_count = 0
            for node_id in component_nodes:
                degree = len([segment_id for segment_id in adjacency[node_id] if segment_id in component])
                if degree > 2:
                    branch_node_count += 1
            style_counts = Counter(segment.style_class for segment in component_segments)
            style_class = "walk_main" if style_counts["walk_main"] > 0 else style_counts.most_common(1)[0][0]
            route_counter_by_page[page_no] += 1
            route_id = f"route_{page_no:03d}_{route_counter_by_page[page_no]:04d}"
            frame_source_kind = frame_source_by_id.get(frame_id)
            confidence, route_reasons = component_confidence(
                component_segments,
                branch_node_count=branch_node_count,
                chain_count=len(chain_points),
                frame_source_kind=frame_source_kind,
            )
            bbox = bbox_union([segment.bbox for segment in component_segments])
            merged_route = MergedRoute(
                route_id=route_id,
                page_no=page_no,
                frame_id=frame_id,
                frame_source_kind=frame_source_kind,
                style_class=style_class,
                chains_pdf=chain_points,
                segment_ids=[segment.segment_id for segment in component_segments],
                segment_count=len(component_segments),
                branch_node_count=branch_node_count,
                chain_count=len(chain_points),
                confidence=confidence,
                needs_manual_review=bool(route_reasons),
                review_reasons=route_reasons,
                component_bbox=bbox,
            )
            merged_routes.append(merged_route)
            for reason in route_reasons:
                add_issue(
                    object_type="route",
                    object_id=route_id,
                    page_no=page_no,
                    frame_id=frame_id,
                    severity="warning",
                    reason=reason,
                    details=f"segment_count={len(component_segments)}, chain_count={len(chain_points)}, branch_nodes={branch_node_count}",
                )

    raw_features: list[dict[str, Any]] = []
    for segment in raw_segments:
        geometry = {
            "type": "LineString",
            "coordinates": to_synthetic_coords(segment.points_pdf),
        }
        properties = {
            "segment_id": segment.segment_id,
            "page_no": segment.page_no,
            "frame_id": segment.frame_id,
            "style_class": segment.style_class,
            "classification": segment.classification,
            "width_pt": round(segment.width_pt, 3),
            "dashes": segment.dashes,
            "length_pt": round(segment.length_pt, 3),
            "confidence": round(segment.confidence, 3),
            "needs_manual_review": segment.needs_manual_review,
            "review_reasons": ",".join(segment.review_reasons),
        }
        raw_features.append(make_geojson_feature(geometry, properties))

    merged_features: list[dict[str, Any]] = []
    for route in merged_routes:
        if len(route.chains_pdf) == 1:
            geometry = {"type": "LineString", "coordinates": to_synthetic_coords(route.chains_pdf[0])}
        else:
            geometry = {"type": "MultiLineString", "coordinates": [to_synthetic_coords(chain) for chain in route.chains_pdf]}
        properties = {
            "route_id": route.route_id,
            "page_no": route.page_no,
            "frame_id": route.frame_id,
            "frame_source_kind": route.frame_source_kind,
            "style_class": route.style_class,
            "segment_count": route.segment_count,
            "chain_count": route.chain_count,
            "branch_node_count": route.branch_node_count,
            "confidence": round(route.confidence, 3),
            "needs_manual_review": route.needs_manual_review,
            "review_reasons": ",".join(route.review_reasons),
            "coordinate_space": "pdf_debug_local",
            "source_pdf": pdf_path.name,
        }
        merged_features.append(make_geojson_feature(geometry, properties))

    route_pages = sorted({route.page_no for route in merged_routes} | {segment.page_no for segment in raw_segments if segment.frame_id})
    page_manifest: list[dict[str, Any]] = []
    for page_no in route_pages:
        page = doc.load_page(page_no - 1)
        image_name = f"page_{page_no:03d}.png"
        pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        image_path = debug_dir / "pages" / image_name
        pix.save(image_path)
        page_manifest.append(page_image_manifest_entry(page_no, page.rect, image_name))

    raw_geojson_path = out_dir / "raw_segments.geojson"
    merged_geojson_path = out_dir / "merged_routes.geojson"
    manual_review_csv_path = out_dir / "manual_review.csv"
    manual_review_json_path = debug_dir / "manual_review.json"
    report_path = out_dir / "step3_report.md"
    log_path = out_dir / "extraction_log.json"

    write_geojson(raw_geojson_path, raw_features)
    write_geojson(merged_geojson_path, merged_features)
    write_manual_review_csv(manual_review_csv_path, manual_review_rows)
    manual_review_json_path.write_text(json.dumps(manual_review_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "raw_segments.geojson").write_text(raw_geojson_path.read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "merged_routes.geojson").write_text(merged_geojson_path.read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "page_manifest.json").write_text(json.dumps(page_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    token = load_env_token(env_path)
    build_html(
        debug_dir=debug_dir,
        config={
            "mapboxToken": token,
            "generatedAt": "step3",
        },
    )

    issue_counter = Counter(row["reason"] for row in manual_review_rows)
    report_lines = [
        "# Step 3 Report",
        "",
        f"- PDF: `{pdf_path.name}`",
        f"- Raw route segments: `{len(raw_segments)}`",
        f"- Merged routes: `{len(merged_routes)}`",
        f"- Manual review issues: `{len(manual_review_rows)}`",
        f"- Pages with route output: `{len(route_pages)}`",
        "",
        "## Diagnostics",
        f"- Missing frame assignment segments: `{issue_counter.get('missing_frame_id', 0)}`",
        f"- Low confidence segments: `{issue_counter.get('low_confidence', 0)}`",
        f"- Branching merged routes: `{issue_counter.get('branching_component', 0)}`",
        f"- Multi-chain merged routes: `{issue_counter.get('multi_chain_component', 0)}`",
        f"- Fallback frame merged routes: `{issue_counter.get('fallback_frame', 0)}`",
        "",
        "## Outputs",
        f"- Raw segments: `{raw_geojson_path}`",
        f"- Merged routes: `{merged_geojson_path}`",
        f"- Manual review CSV: `{manual_review_csv_path}`",
        f"- Mapbox debug HTML: `{debug_dir / 'index.html'}`",
        "",
        "## Notes",
        "- Mapbox debug is rendered in synthetic PDF-local coordinates, not geographic coordinates.",
        "- If the overlay looks wrong in the debug HTML, the cause should be visible via the manual review list and frame source metadata.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = {
        "pdf_path": str(pdf_path),
        "raw_segment_count": len(raw_segments),
        "merged_route_count": len(merged_routes),
        "manual_review_count": len(manual_review_rows),
        "pages_with_routes": len(route_pages),
        "issue_counts": issue_counter,
        "outputs": {
            "raw_segments_geojson": str(raw_geojson_path),
            "merged_routes_geojson": str(merged_geojson_path),
            "manual_review_csv": str(manual_review_csv_path),
            "report_md": str(report_path),
            "mapbox_debug_dir": str(debug_dir),
        },
    }
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: dict(value)), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize route candidates into raw and merged GeoJSON with Mapbox debug output.")
    parser.add_argument("pdf", type=Path, help="Path to the target PDF")
    parser.add_argument("--step2-dir", type=Path, default=Path("artifacts/step2"), help="Directory containing Step 2 outputs")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/step3"), help="Output directory")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Environment file used to inject an optional Mapbox token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_step3(
        pdf_path=args.pdf,
        step2_dir=args.step2_dir,
        out_dir=args.out_dir,
        env_path=args.env_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: dict(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
