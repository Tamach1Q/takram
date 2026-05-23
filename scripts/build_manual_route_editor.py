#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


DEFAULT_CENTER = [133.5, 33.8]
DEFAULT_ZOOM = 8.5
RENDER_SCALE = 2.0
EDITOR_VERSION = "manual-route-editor-v1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    return read_csv_rows(path)


def read_env_vars(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def read_mapbox_token(env_path: Path) -> str:
    return read_env_vars(env_path).get("MAPBOX_ACCESS_TOKEN", "")


def takram_image_dir(env_path: Path) -> Path:
    env_vars = read_env_vars(env_path)
    raw = env_vars.get("TAKRAM_IMAGE_DIR", "../takram-image")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (env_path.parent / candidate).resolve()
    return candidate


def parse_bbox_pt(text: str) -> dict[str, float]:
    x0, y0, x1, y1 = [float(value) for value in str(text).split(";")]
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def rounded_bbox(row: dict[str, Any]) -> dict[str, float]:
    bbox = parse_bbox_pt(row["bbox_pt"])
    return {
        "x0": round(bbox["x0"], 3),
        "y0": round(bbox["y0"], 3),
        "x1": round(bbox["x1"], 3),
        "y1": round(bbox["y1"], 3),
    }


def bbox_area_pt2(bbox: dict[str, float]) -> float:
    return max(0.0, bbox["x1"] - bbox["x0"]) * max(0.0, bbox["y1"] - bbox["y0"])


def point_in_bbox(x: float, y: float, bbox: dict[str, float], *, margin: float = 0.0) -> bool:
    return (
        bbox["x0"] - margin <= x <= bbox["x1"] + margin
        and bbox["y0"] - margin <= y <= bbox["y1"] + margin
    )


def route_features_by_panel(paths: list[Path]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        data = read_geojson(path)
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            page_no = props.get("page_no")
            panel_id = props.get("georef_panel_id")
            if page_no in (None, "") or not panel_id:
                continue
            grouped[(int(page_no), str(panel_id))].append(feature)
    return grouped


def panel_geojson_by_key(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return grouped
    data = read_geojson(path)
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        page_no = props.get("page_no")
        panel_id = props.get("georef_panel_id")
        if page_no in (None, "") or not panel_id:
            continue
        grouped[(int(page_no), str(panel_id))] = feature
    return grouped


def polygon_coords_from_feature(feature: dict[str, Any] | None) -> list[list[float]]:
    if not feature:
        return []
    geometry = feature.get("geometry", {})
    if geometry.get("type") != "Polygon":
        return []
    rings = geometry.get("coordinates", [])
    if not rings:
        return []
    return [[round(float(coord[0]), 3), round(float(coord[1]), 3)] for coord in rings[0]]


def panel_key_from_mapping(data: dict[str, Any]) -> tuple[int, str] | None:
    page_no = data.get("page_no")
    panel_id = data.get("georef_panel_id") or data.get("frame_id")
    if page_no in (None, "") or not panel_id:
        return None
    return (int(page_no), str(panel_id))


def saved_georef_by_panel(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return result
    for json_path in sorted(path.glob("page_*_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        key = panel_key_from_mapping(data)
        if key is None:
            continue
        data["__path__"] = str(json_path)
        result[key] = data
    return result


def render_panel_images(
    *,
    doc: fitz.Document,
    page_no: int,
    panel_row: dict[str, Any],
    render_scale: float,
) -> tuple[Image.Image, Image.Image]:
    page = doc[page_no - 1]
    bbox = parse_bbox_pt(panel_row["bbox_pt"])
    clip = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
    pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), clip=clip, alpha=False)
    rgb = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    rgba = rgb.convert("RGBA")
    src = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, _ = src[x, y]
            if r >= 180 and g < 110 and b < 110:
                src[x, y] = (r, g, b, 255)
            else:
                src[x, y] = (0, 0, 0, 0)
    return rgb, rgba


def mean_center(points: list[tuple[float, float]]) -> list[float]:
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return [round(lon, 7), round(lat, 7)]


def zoom_from_extent(points: list[tuple[float, float]], image_width_px: int, image_height_px: int) -> float | None:
    if len(points) < 2:
        return None
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    center_lat = sum(lats) / len(lats)
    lon_span_m = (max(lons) - min(lons)) * 111320.0 * max(0.1, math.cos(math.radians(center_lat)))
    lat_span_m = (max(lats) - min(lats)) * 110540.0
    span_m = max(lon_span_m, lat_span_m)
    if span_m <= 1e-6:
        return None
    target_px = max(320.0, max(image_width_px, image_height_px) * 1.35)
    meters_per_pixel = span_m / target_px
    world_mpp = 156543.03392 * max(0.1, math.cos(math.radians(center_lat)))
    return math.log2(world_mpp / meters_per_pixel)


def corners_dict_to_points(corners: dict[str, list[float]] | None) -> list[tuple[float, float]]:
    if not corners:
        return []
    points: list[tuple[float, float]] = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        if name not in corners:
            continue
        lon, lat = corners[name]
        points.append((float(lon), float(lat)))
    return points


def serializable_manual_geo_points(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not data:
        return []
    points: list[dict[str, Any]] = []
    for index, row in enumerate(data.get("gcps", []), start=1):
        if row.get("longitude") in (None, "") or row.get("latitude") in (None, ""):
            continue
        points.append(
            {
                "index": index,
                "name": row.get("name") or row.get("role") or f"manual_{index}",
                "role": row.get("role", ""),
                "longitude": round(float(row["longitude"]), 7),
                "latitude": round(float(row["latitude"]), 7),
            }
        )
    return points


def serializable_auto_geo_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("longitude") in (None, "") or row.get("latitude") in (None, ""):
            continue
        points.append(
            {
                "index": index,
                "name": row.get("source_name_text") or row.get("gazetteer_name_short") or row.get("gcp_id") or f"auto_{index}",
                "role": row.get("source_kind", ""),
                "longitude": round(float(row["longitude"]), 7),
                "latitude": round(float(row["latitude"]), 7),
            }
        )
    return points


def serializable_temple_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("longitude") in (None, "") or row.get("latitude") in (None, ""):
            continue
        key = (
            str(row.get("temple_group", "")),
            str(row.get("temple_no", "")),
            str(row.get("longitude", "")),
            str(row.get("latitude", "")),
        )
        confidence = float(row.get("confidence") or 0.0)
        current = best_by_key.get(key)
        if current is not None and float(current.get("confidence") or 0.0) >= confidence:
            continue
        temple_group = str(row.get("temple_group", ""))
        temple_no = str(row.get("temple_no", ""))
        no_label = f"別{temple_no}" if temple_group == "bekkaku" else temple_no
        best_by_key[key] = {
            "gcp_id": row.get("gcp_id", ""),
            "temple_group": temple_group,
            "temple_no": temple_no,
            "temple_no_label": no_label,
            "name_full": row.get("gazetteer_name_full") or row.get("source_name_text") or "",
            "name_short": row.get("gazetteer_name_short") or row.get("source_name_text") or "",
            "source_name_text": row.get("source_name_text", ""),
            "longitude": round(float(row["longitude"]), 7),
            "latitude": round(float(row["latitude"]), 7),
            "confidence": confidence,
            "needs_manual_review": str(row.get("needs_manual_review", "")).strip().lower() == "true",
            "review_reasons": row.get("review_reasons", ""),
            "source_kind": row.get("source_kind", ""),
        }
    return sorted(
        best_by_key.values(),
        key=lambda row: (row["temple_group"], int(row["temple_no"] or 0), row["name_short"]),
    )


def gcp_rows_by_panel(
    panel_rows: list[dict[str, Any]],
    gcp_rows: list[dict[str, Any]],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    panels_by_page: dict[int, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    for panel_row in panel_rows:
        panels_by_page[int(panel_row["page_no"])].append((str(panel_row["georef_panel_id"]), parse_bbox_pt(panel_row["bbox_pt"])))

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gcp_rows:
        if row.get("page_no") in (None, ""):
            continue
        page_no = int(row["page_no"])
        x_value = row.get("pdf_anchor_x_pt") or row.get("pdf_label_x_pt")
        y_value = row.get("pdf_anchor_y_pt") or row.get("pdf_label_y_pt")
        if x_value in (None, "") or y_value in (None, ""):
            continue
        x = float(x_value)
        y = float(y_value)
        for panel_id, bbox in panels_by_page.get(page_no, []):
            if point_in_bbox(x, y, bbox, margin=0.5):
                grouped[(page_no, panel_id)].append(row)
                break
    return grouped


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def choose_initial_view(saved_row: dict[str, Any] | None, image_width_px: int, image_height_px: int) -> tuple[list[float], float]:
    if saved_row and saved_row.get("corners_lonlat"):
        points = corners_dict_to_points(saved_row["corners_lonlat"])
        if points:
            center = mean_center(points)
            zoom = zoom_from_extent(points, image_width_px, image_height_px)
            if zoom is not None:
                return center, round(max(7.0, min(17.0, zoom)), 2)
    return DEFAULT_CENTER[:], DEFAULT_ZOOM


def build_html(config: dict[str, Any]) -> str:
    template = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual Route Editor</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
  <style>
    :root {
      --bg: #f5f3ee;
      --surface: #ffffff;
      --border: #d7d2c8;
      --text: #111827;
      --muted: #4b5563;
      --accent: #b91c1c;
      --accentBlue: #2563eb;
      --accentGray: #9ca3af;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); }
    #app { display: grid; grid-template-rows: auto auto 1fr; height: 100%; min-height: 0; }
    #toolbar {
      display: grid;
      grid-template-columns: auto minmax(320px, 1fr) auto repeat(8, auto);
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.94);
    }
    #statusbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.9);
    }
    .toolbarLabel { font-size: 12px; font-weight: 800; color: var(--muted); }
    select, button, textarea {
      font: inherit;
      border: 1px solid rgba(15,23,42,0.14);
      border-radius: 10px;
      background: white;
    }
    select, button { min-height: 34px; padding: 6px 10px; }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: transparent; }
    #panelCount {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(15,23,42,0.1);
      background: rgba(255,255,255,0.92);
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
    #workspace { display: grid; grid-template-columns: 45fr 55fr; min-height: 0; }
    .pane {
      position: relative;
      min-width: 0;
      min-height: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,244,239,0.98) 100%);
      border-right: 1px solid var(--border);
    }
    .pane:last-child { border-right: 0; }
    .paneHeader {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      z-index: 6;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.72) 100%);
      border-bottom: 1px solid rgba(15,23,42,0.08);
      backdrop-filter: blur(4px);
    }
    .paneTitle { font-size: 13px; font-weight: 900; letter-spacing: 0.04em; }
    .paneControls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    #pdfCanvas { position: absolute; top: 0; left: 0; display: block; cursor: grab; touch-action: none; }
    #pdfCanvas.dragging { cursor: grabbing; }
    #map { position: absolute; inset: 0; }
    #pdfDebug, #mapDebug {
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 6;
      margin: 0;
      padding: 10px 12px;
      width: min(360px, calc(100% - 24px));
      background: rgba(15,23,42,0.84);
      color: #e2e8f0;
      border-radius: 12px;
      font-size: 11px;
      line-height: 1.45;
      white-space: pre-wrap;
      pointer-events: none;
    }
    #notice { font-size: 12px; color: var(--muted); white-space: pre-wrap; }
    #metrics { display: flex; justify-content: flex-end; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
    .metricBadge {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 12px;
      border-radius: 999px;
      background: rgba(37,99,235,0.12);
      color: var(--accentBlue);
      font-size: 12px;
      font-weight: 900;
    }
    #notesPanel {
      position: absolute;
      right: 12px;
      bottom: 12px;
      z-index: 6;
      width: min(360px, calc(100% - 24px));
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(15,23,42,0.12);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.08);
      padding: 12px;
    }
    #notesPanel h3 { margin: 0 0 8px 0; font-size: 13px; }
    #panelNotes {
      width: 100%;
      min-height: 86px;
      resize: vertical;
      padding: 10px 12px;
      line-height: 1.45;
    }
    #routeList {
      margin-top: 10px;
      display: grid;
      gap: 8px;
      max-height: 180px;
      overflow: auto;
    }
    .routeRow {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid rgba(15,23,42,0.1);
      background: rgba(248,250,252,0.96);
      cursor: pointer;
    }
    .routeRow.active {
      border-color: rgba(37,99,235,0.32);
      background: rgba(239,246,255,0.98);
    }
    .routeMeta { font-size: 11px; color: var(--muted); white-space: pre-wrap; }
    .routeDelete {
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      border: 1px solid rgba(220,38,38,0.22);
      background: white;
      color: #b91c1c;
      font-size: 11px;
      font-weight: 800;
    }
    .help {
      position: absolute;
      top: 72px;
      left: 12px;
      z-index: 6;
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(15,23,42,0.1);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 11px;
      line-height: 1.5;
      color: var(--muted);
      max-width: 320px;
    }
    @media (max-width: 1200px) {
      #toolbar { grid-template-columns: 1fr 1fr; }
      #workspace { grid-template-columns: 1fr; grid-template-rows: 40vh 1fr; }
      .pane { border-right: 0; border-bottom: 1px solid var(--border); }
      .pane:last-child { border-bottom: 0; }
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="toolbar">
      <div class="toolbarLabel">Panel</div>
      <select id="panelSelect"></select>
      <div id="panelCount"></div>
      <button id="prevButton">Previous Panel</button>
      <button id="nextButton">Next Panel</button>
      <button id="bindDirButton">Bind Output Dir</button>
      <button id="newRouteButton">New Route</button>
      <button id="undoButton">Undo</button>
      <button id="saveButton" class="primary">Save</button>
      <button id="toggleAllRoutesButton">All Routes ON</button>
      <button id="toggleReferenceButton">Reference OFF</button>
      <button id="toggleTempleButton">Temple ON</button>
    </div>
    <div id="statusbar">
      <div id="stepText">右の Mapbox をクリックしてルートを描きます</div>
      <div id="metrics"></div>
      <div id="notice"></div>
    </div>
    <div id="workspace">
      <section class="pane" id="pdfPane">
        <div class="paneHeader">
          <div class="paneTitle">PDF Reference</div>
          <div class="paneControls">
            <select id="pdfDisplaySelect">
              <option value="map">元地図</option>
              <option value="rawMask">Raw red mask</option>
              <option value="accepted">Accepted routes reference</option>
              <option value="mapAccepted">元地図 + reference</option>
            </select>
            <button id="pdfFitAllButton">Fit All</button>
            <button id="pdfFitHeightButton">Fit Height</button>
            <button id="pdfFitWidthButton">Fit Width</button>
            <button id="pdfZoomOutButton">-</button>
            <button id="pdfZoomInButton">+</button>
          </div>
        </div>
        <canvas id="pdfCanvas"></canvas>
        <div class="help">
          左は参照専用です。<br/>
          Fit All が初期値です。ドラッグで pan、wheel / trackpad で zoom できます。<br/>
          Accepted routes reference は壊れている可能性があるため参考表示です。
        </div>
        <pre id="pdfDebug"></pre>
      </section>
      <section class="pane" id="mapPane">
        <div class="paneHeader">
          <div class="paneTitle">Mapbox Manual Trace</div>
          <div class="paneControls">
            <span class="toolbarLabel">Mode</span>
            <select id="editModeSelect">
              <option value="edit">編集</option>
              <option value="delete">削除</option>
            </select>
          </div>
        </div>
        <div id="map"></div>
        <div class="help">
          click: 点追加 / segment click: 中間点追加 / drag vertex: 移動<br/>
          Shift+click: 新しいLineString / double click: 現在のLineString確定<br/>
          Z: Undo / Backspace: 最後の点削除 / S: 保存 / N,P: panel移動<br/>
          直線は少ない点でよい。曲がり角、分岐、橋、寺入口では必ず点を打つ。<br/>
          カーブでは赤線が道路から外れない程度に点を追加する。山道や細い道はやや細かく打つ。<br/>
          目安: 直線100〜300m / カーブ30〜80m / 山道10〜50m
        </div>
        <pre id="mapDebug"></pre>
        <div id="notesPanel">
          <h3>Panel Notes</h3>
          <textarea id="panelNotes" placeholder="OSMに道なし / 山道 / 要確認 など"></textarea>
          <div id="routeList"></div>
        </div>
      </section>
    </div>
  </div>

  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <script>
    window.__MANUAL_ROUTE_EDITOR_CONFIG__ = __CONFIG_JSON__;
  </script>
  <script>
    const config = window.__MANUAL_ROUTE_EDITOR_CONFIG__;
    const ui = {
      panelSelect: document.getElementById('panelSelect'),
      panelCount: document.getElementById('panelCount'),
      prevButton: document.getElementById('prevButton'),
      nextButton: document.getElementById('nextButton'),
      bindDirButton: document.getElementById('bindDirButton'),
      newRouteButton: document.getElementById('newRouteButton'),
      undoButton: document.getElementById('undoButton'),
      saveButton: document.getElementById('saveButton'),
      toggleAllRoutesButton: document.getElementById('toggleAllRoutesButton'),
      toggleReferenceButton: document.getElementById('toggleReferenceButton'),
      toggleTempleButton: document.getElementById('toggleTempleButton'),
      pdfDisplaySelect: document.getElementById('pdfDisplaySelect'),
      pdfFitAllButton: document.getElementById('pdfFitAllButton'),
      pdfFitHeightButton: document.getElementById('pdfFitHeightButton'),
      pdfFitWidthButton: document.getElementById('pdfFitWidthButton'),
      pdfZoomOutButton: document.getElementById('pdfZoomOutButton'),
      pdfZoomInButton: document.getElementById('pdfZoomInButton'),
      editModeSelect: document.getElementById('editModeSelect'),
      stepText: document.getElementById('stepText'),
      metrics: document.getElementById('metrics'),
      notice: document.getElementById('notice'),
      pdfCanvas: document.getElementById('pdfCanvas'),
      pdfPane: document.getElementById('pdfPane'),
      pdfDebug: document.getElementById('pdfDebug'),
      mapDebug: document.getElementById('mapDebug'),
      panelNotes: document.getElementById('panelNotes'),
      routeList: document.getElementById('routeList'),
    };
    const pdfCtx = ui.pdfCanvas.getContext('2d');
    const imageCache = new Map();
    const panelViewState = new Map();
    let currentPanelId = '';
    let map = null;
    let mapReady = false;
    let vertexMarkers = [];
    let outputDirHandle = null;
    let allRoutesVisible = true;
    let referenceVisible = false;
    let templeVisible = true;
    const historyByPanel = new Map();
    const state = {
      panels: {},
      activeRouteIdByPanel: {},
      storageUpdatedAt: null,
      lastMapView: null,
    };

    function ensurePanelState(panelId) {
      if (!state.panels[panelId]) {
        state.panels[panelId] = { routes: [], notes: '', updated_at: null };
      }
      return state.panels[panelId];
    }

    function panelById(panelId) {
      return config.panels.find((panel) => panel.georef_panel_id === panelId) || null;
    }

    function sortedPanels() {
      return [...config.panels].sort((a, b) => (a.page_no - b.page_no) || a.georef_panel_id.localeCompare(b.georef_panel_id));
    }

    function getPanelViewState(panel) {
      if (!panelViewState.has(panel.georef_panel_id)) {
        panelViewState.set(panel.georef_panel_id, {
          fitMode: 'all',
          zoom: 1,
          panX: 0,
          panY: 0,
          fitApplied: false,
          dragState: null,
        });
      }
      return panelViewState.get(panel.georef_panel_id);
    }

    function loadImage(url) {
      if (imageCache.has(url)) return imageCache.get(url);
      const promise = new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`画像を読み込めません: ${url}`));
        image.src = url;
      });
      imageCache.set(url, promise);
      return promise;
    }

    function panelRoutes(panelId) {
      return ensurePanelState(panelId).routes;
    }

    function panelNotes(panelId) {
      return ensurePanelState(panelId).notes || '';
    }

    function manualGeoPoints(panel) {
      return panel?.manual_geo_points || [];
    }

    function autoGeoPoints(panel) {
      return panel?.auto_geo_points || [];
    }

    function templePoints(panel) {
      return panel?.temple_points || [];
    }

    function activeRouteId(panelId = currentPanelId) {
      return state.activeRouteIdByPanel[panelId] || null;
    }

    function setActiveRouteId(panelId, routeId) {
      state.activeRouteIdByPanel[panelId] = routeId || null;
    }

    function currentEditMode() {
      return ui.editModeSelect.value;
    }

    function clone(obj) {
      return JSON.parse(JSON.stringify(obj));
    }

    function pushHistory(panelId) {
      const history = historyByPanel.get(panelId) || [];
      history.push(clone(ensurePanelState(panelId)));
      if (history.length > 50) history.shift();
      historyByPanel.set(panelId, history);
    }

    function undoPanel(panelId) {
      const history = historyByPanel.get(panelId) || [];
      if (!history.length) return;
      state.panels[panelId] = history.pop();
      touchPanel(panelId, false);
      renderAll();
    }

    function touchPanel(panelId, save = true) {
      const panelState = ensurePanelState(panelId);
      panelState.updated_at = new Date().toISOString();
      if (save) scheduleAutosave();
    }

    function currentPanelStatus(panelId) {
      const routes = panelRoutes(panelId);
      if (!routes.length) return 'empty';
      if (routes.every((route) => route.status === 'done')) return 'done';
      if (routes.some((route) => route.status === 'reviewed')) return 'reviewed';
      return 'draft';
    }

    function haversineMeters(a, b) {
      const rad = Math.PI / 180;
      const dLat = (b[1] - a[1]) * rad;
      const dLon = (b[0] - a[0]) * rad;
      const lat1 = a[1] * rad;
      const lat2 = b[1] * rad;
      const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
      return 2 * 6378137 * Math.asin(Math.min(1, Math.sqrt(h)));
    }

    function routeLengthMeters(coords) {
      let total = 0;
      for (let index = 0; index < coords.length - 1; index += 1) {
        total += haversineMeters(coords[index], coords[index + 1]);
      }
      return total;
    }

    function routeVertexCount(panelId) {
      return panelRoutes(panelId).reduce((sum, route) => sum + route.coordinates.length, 0);
    }

    function panelLengthMeters(panelId) {
      return panelRoutes(panelId).reduce((sum, route) => sum + routeLengthMeters(route.coordinates), 0);
    }

    function buildRouteFeature(panel, route) {
      return {
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: route.coordinates },
        properties: {
          source: 'manual_mapbox_trace',
          route_id: route.route_id,
          page_no: panel.page_no,
          georef_panel_id: panel.georef_panel_id,
          panel_type: panel.panel_type,
          status: route.status || 'draft',
          length_m: Number(routeLengthMeters(route.coordinates).toFixed(3)),
          vertex_count: route.coordinates.length,
          created_at: route.created_at,
          updated_at: route.updated_at || route.created_at,
          notes: route.notes || '',
          editor_version: config.editor_version,
        },
      };
    }

    function buildManualRoutesGeojson() {
      const features = [];
      sortedPanels().forEach((panel) => {
        panelRoutes(panel.georef_panel_id).forEach((route) => {
          if (route.coordinates.length >= 2) {
            features.push(buildRouteFeature(panel, route));
          }
        });
      });
      return { type: 'FeatureCollection', features };
    }

    function buildAutosavePayload() {
      return {
        editor_version: config.editor_version,
        updated_at: new Date().toISOString(),
        panels: state.panels,
        active_route_id_by_panel: state.activeRouteIdByPanel,
        last_map_view: state.lastMapView,
      };
    }

    function emptyRoute(panel) {
      const timestamp = Date.now();
      return {
        route_id: `${panel.georef_panel_id}_route_${timestamp}`,
        coordinates: [],
        status: 'draft',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        notes: '',
      };
    }

    function startNewRoute(panelId, seedCoord = null) {
      const panel = panelById(panelId);
      const panelState = ensurePanelState(panelId);
      const route = emptyRoute(panel);
      if (seedCoord) route.coordinates.push(seedCoord);
      pushHistory(panelId);
      panelState.routes.push(route);
      setActiveRouteId(panelId, route.route_id);
      touchPanel(panelId);
      return route;
    }

    function activeRoute(panelId = currentPanelId) {
      const routeId = activeRouteId(panelId);
      if (!routeId) return null;
      return panelRoutes(panelId).find((route) => route.route_id === routeId) || null;
    }

    function removeRoute(panelId, routeId) {
      pushHistory(panelId);
      const panelState = ensurePanelState(panelId);
      panelState.routes = panelState.routes.filter((route) => route.route_id !== routeId);
      if (activeRouteId(panelId) === routeId) {
        setActiveRouteId(panelId, panelState.routes.at(-1)?.route_id || null);
      }
      touchPanel(panelId);
      renderAll();
    }

    function finalizeActiveRoute(panelId = currentPanelId) {
      const route = activeRoute(panelId);
      if (!route || route.coordinates.length < 2) return;
      route.updated_at = new Date().toISOString();
      setActiveRouteId(panelId, null);
      touchPanel(panelId);
      renderAll();
    }

    function computeRouteCoordStats(panel) {
      const features = panel.reference_route_geojson?.features || [];
      const coords = [];
      features.forEach((feature) => {
        const geometry = feature.geometry || {};
        const lines = geometry.type === 'LineString' ? [geometry.coordinates || []] : (geometry.coordinates || []);
        lines.forEach((line) => line.forEach((coord) => coords.push(coord)));
      });
      if (!coords.length) {
        return { featureCount: 0, bbox: null };
      }
      const xs = coords.map((coord) => Number(coord[0]));
      const ys = coords.map((coord) => Number(coord[1]));
      return {
        featureCount: features.length,
        bbox: { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) },
      };
    }

    function pagePtToImagePx(panel, coord) {
      const bbox = panel.pdf_bbox;
      const scaleX = panel.image_width_px / Math.max(1, bbox.x1 - bbox.x0);
      const scaleY = panel.image_height_px / Math.max(1, bbox.y1 - bbox.y0);
      return [(coord[0] - bbox.x0) * scaleX, (coord[1] - bbox.y0) * scaleY];
    }

    function drawReferenceRoutes(panel, layout) {
      const features = panel.reference_route_geojson?.features || [];
      pdfCtx.save();
      pdfCtx.beginPath();
      features.forEach((feature) => {
        const geometry = feature.geometry || {};
        const lines = geometry.type === 'LineString' ? [geometry.coordinates || []] : (geometry.coordinates || []);
        lines.forEach((line) => {
          line.forEach((coord, index) => {
            const imagePoint = pagePtToImagePx(panel, coord);
            const x = layout.x + imagePoint[0] * layout.scale;
            const y = layout.y + imagePoint[1] * layout.scale;
            if (index === 0) pdfCtx.moveTo(x, y);
            else pdfCtx.lineTo(x, y);
          });
        });
      });
      pdfCtx.strokeStyle = 'rgba(220,38,38,0.96)';
      pdfCtx.lineWidth = 3.2;
      pdfCtx.lineCap = 'round';
      pdfCtx.lineJoin = 'round';
      pdfCtx.stroke();
      pdfCtx.restore();
    }

    function resizePdfCanvas() {
      const header = ui.pdfPane.querySelector('.paneHeader');
      const paneRect = ui.pdfPane.getBoundingClientRect();
      const headerHeight = header ? header.getBoundingClientRect().height : 0;
      const width = Math.max(1, Math.round(paneRect.width));
      const height = Math.max(1, Math.round(paneRect.height));
      ui.pdfCanvas.style.top = `${Math.round(headerHeight)}px`;
      ui.pdfCanvas.style.width = `${width}px`;
      ui.pdfCanvas.style.height = `${height - headerHeight}px`;
      const rect = ui.pdfCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      ui.pdfCanvas.width = Math.round(rect.width * dpr);
      ui.pdfCanvas.height = Math.round(rect.height * dpr);
      pdfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return rect;
    }

    function pdfBaseScale(panel, rect, fitMode = 'all') {
      const widthScale = rect.width / panel.image_width_px;
      const heightScale = rect.height / panel.image_height_px;
      if (fitMode === 'width') return widthScale;
      if (fitMode === 'height') return heightScale;
      return Math.min(widthScale, heightScale);
    }

    function applyPdfFit(panel, rect, view, fitMode = 'all') {
      const baseScale = pdfBaseScale(panel, rect, fitMode);
      const scaledWidth = panel.image_width_px * baseScale;
      const scaledHeight = panel.image_height_px * baseScale;
      view.fitMode = fitMode;
      view.zoom = 1;
      view.panX = (rect.width - scaledWidth) / 2;
      view.panY = (rect.height - scaledHeight) / 2;
      view.fitApplied = true;
    }

    function currentPdfLayout(panel, rect, view) {
      if (!view.fitApplied) applyPdfFit(panel, rect, view, view.fitMode || 'all');
      const baseScale = pdfBaseScale(panel, rect, view.fitMode || 'all');
      const scale = baseScale * view.zoom;
      return {
        x: view.panX,
        y: view.panY,
        scale,
        width: panel.image_width_px * scale,
        height: panel.image_height_px * scale,
      };
    }

    function zoomPdfAtPoint(panel, view, rect, pointerX, pointerY, factor) {
      const layout = currentPdfLayout(panel, rect, view);
      const imageX = (pointerX - layout.x) / layout.scale;
      const imageY = (pointerY - layout.y) / layout.scale;
      view.zoom = Math.max(0.25, Math.min(10, view.zoom * factor));
      const newScale = pdfBaseScale(panel, rect, view.fitMode || 'all') * view.zoom;
      view.panX = pointerX - (imageX * newScale);
      view.panY = pointerY - (imageY * newScale);
      view.fitApplied = true;
    }

    async function drawPdfPane() {
      const panel = panelById(currentPanelId);
      const view = getPanelViewState(panel);
      const rect = resizePdfCanvas();
      pdfCtx.clearRect(0, 0, rect.width, rect.height);
      pdfCtx.fillStyle = '#ebe7df';
      pdfCtx.fillRect(0, 0, rect.width, rect.height);
      const layout = currentPdfLayout(panel, rect, view);
      const mode = ui.pdfDisplaySelect.value;
      const showMap = mode === 'map' || mode === 'mapAccepted';
      const showMask = mode === 'rawMask';
      const showReference = mode === 'accepted' || mode === 'mapAccepted';
      if (showMap) {
        const image = await loadImage(panel.image_path);
        pdfCtx.drawImage(image, layout.x, layout.y, layout.width, layout.height);
      }
      if (showMask) {
        const image = await loadImage(panel.redlines_path);
        pdfCtx.globalAlpha = 0.48;
        pdfCtx.drawImage(image, layout.x, layout.y, layout.width, layout.height);
        pdfCtx.globalAlpha = 1;
      }
      if (showReference) {
        if (showMap && !panel.reference_route_geojson.features.length) {
          const image = await loadImage(panel.image_path);
          pdfCtx.drawImage(image, layout.x, layout.y, layout.width, layout.height);
        }
        drawReferenceRoutes(panel, layout);
      }
      pdfCtx.strokeStyle = 'rgba(37,99,235,0.95)';
      pdfCtx.lineWidth = 2.5;
      pdfCtx.strokeRect(layout.x, layout.y, layout.width, layout.height);
      const stats = computeRouteCoordStats(panel);
      ui.pdfDebug.textContent = [
        `page_no: ${panel.page_no}`,
        `georef_panel_id: ${panel.georef_panel_id}`,
        `panel_type: ${panel.panel_type}`,
        `route_count: ${panel.route_count}`,
        `fit_mode: ${view.fitMode || 'all'}`,
        `zoom: ${(view.zoom * 100).toFixed(0)}%`,
        `reference_feature_count: ${stats.featureCount}`,
        `reference_bbox: ${stats.bbox ? `${stats.bbox.x0.toFixed(2)}, ${stats.bbox.y0.toFixed(2)}, ${stats.bbox.x1.toFixed(2)}, ${stats.bbox.y1.toFixed(2)}` : '-'}`,
        `pdf_panel_bbox: ${panel.pdf_bbox.x0.toFixed(2)}, ${panel.pdf_bbox.y0.toFixed(2)}, ${panel.pdf_bbox.x1.toFixed(2)}, ${panel.pdf_bbox.y1.toFixed(2)}`,
      ].join('\\n');
    }

    function computePanelSummary(panelId) {
      const status = currentPanelStatus(panelId);
      return {
        status,
        vertexCount: routeVertexCount(panelId),
        lengthKm: panelLengthMeters(panelId) / 1000,
      };
    }

    function populatePanelSelect() {
      const panels = sortedPanels();
      ui.panelSelect.innerHTML = '';
      panels.forEach((panel, index) => {
        const summary = computePanelSummary(panel.georef_panel_id);
        const option = document.createElement('option');
        option.value = panel.georef_panel_id;
        option.textContent = `${index + 1}/${panels.length}  p${panel.page_no} / ${panel.georef_panel_id} / ${panel.panel_type} / ref ${panel.route_count} / ${summary.status} / v${summary.vertexCount} / ${summary.lengthKm.toFixed(2)}km`;
        ui.panelSelect.appendChild(option);
      });
      ui.panelCount.textContent = `${panels.length} panels`;
      if (!currentPanelId && panels.length) currentPanelId = panels[0].georef_panel_id;
      ui.panelSelect.value = currentPanelId;
    }

    function updateStepText() {
      const panel = panelById(currentPanelId);
      const summary = computePanelSummary(currentPanelId);
      const route = activeRoute();
      const currentLength = route ? routeLengthMeters(route.coordinates) / 1000 : 0;
      ui.stepText.textContent = `Mapbox上をクリックして ${panel.georef_panel_id} の徒歩ルートを描きます`;
      ui.metrics.innerHTML = `
        <span class="metricBadge">status: ${summary.status}</span>
        <span><strong>vertex</strong>: ${summary.vertexCount}</span>
        <span><strong>panel km</strong>: ${summary.lengthKm.toFixed(2)}</span>
        <span><strong>current km</strong>: ${currentLength.toFixed(2)}</span>
        <span><strong>temples</strong>: ${templePoints(panel).length}</span>
      `;
      ui.notice.textContent = outputDirHandle
        ? 'autosave: output dir に書き込みます'
        : 'autosave: localStorage / Save時は dir未バインドならダウンロード';
    }

    function buildAllManualRoutesSourceData() {
      const features = [];
      sortedPanels().forEach((panel) => {
        panelRoutes(panel.georef_panel_id)
          .filter((route) => route.coordinates.length >= 2)
          .forEach((route) => features.push(buildRouteFeature(panel, route)));
      });
      return { type: 'FeatureCollection', features };
    }

    function buildCurrentPanelRoutesSourceData(panelId = currentPanelId) {
      const panel = panelById(panelId);
      return {
        type: 'FeatureCollection',
        features: panelRoutes(panelId)
          .filter((route) => route.coordinates.length >= 2)
          .map((route) => buildRouteFeature(panel, route)),
      };
    }

    function ensureMapSource(id, data) {
      if (!map.getSource(id)) {
        map.addSource(id, { type: 'geojson', data });
      } else {
        map.getSource(id).setData(data);
      }
    }

    function lineFeatureForActiveRoute(route) {
      if (!route || route.coordinates.length < 2) return { type: 'FeatureCollection', features: [] };
      return {
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: route.coordinates },
          properties: { route_id: route.route_id, active: true },
        }],
      };
    }

    function projectiveSolve(rows, values, columns) {
      const ata = Array.from({ length: columns }, () => Array(columns).fill(0));
      const atb = Array(columns).fill(0);
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex];
        for (let i = 0; i < columns; i += 1) {
          atb[i] += row[i] * values[rowIndex];
          for (let j = 0; j < columns; j += 1) ata[i][j] += row[i] * row[j];
        }
      }
      const matrix = ata.map((row, index) => [...row, atb[index]]);
      for (let col = 0; col < columns; col += 1) {
        let pivot = col;
        for (let row = col + 1; row < columns; row += 1) {
          if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
        }
        if (Math.abs(matrix[pivot][col]) < 1e-9) return null;
        if (pivot !== col) [matrix[pivot], matrix[col]] = [matrix[col], matrix[pivot]];
        const divisor = matrix[col][col];
        for (let k = col; k <= columns; k += 1) matrix[col][k] /= divisor;
        for (let row = 0; row < columns; row += 1) {
          if (row === col) continue;
          const factor = matrix[row][col];
          for (let k = col; k <= columns; k += 1) matrix[row][k] -= factor * matrix[col][k];
        }
      }
      return matrix.map((row) => row[columns]);
    }

    function buildReferenceGeojsonForMap(panel) {
      if (!panel.saved_corners || !(panel.reference_route_geojson?.features || []).length) {
        return { type: 'FeatureCollection', features: [] };
      }
      const src = [
        [0, 0],
        [panel.image_width_px, 0],
        [panel.image_width_px, panel.image_height_px],
        [0, panel.image_height_px],
      ];
      const dst = [
        panel.saved_corners.top_left,
        panel.saved_corners.top_right,
        panel.saved_corners.bottom_right,
        panel.saved_corners.bottom_left,
      ];
      const rows = [];
      const values = [];
      for (let index = 0; index < src.length; index += 1) {
        const [x, y] = src[index];
        const [u, v] = dst[index];
        rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); values.push(u);
        rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]); values.push(v);
      }
      const params = projectiveSolve(rows, values, 8);
      if (!params) return { type: 'FeatureCollection', features: [] };
      const project = (point) => {
        const [h11, h12, h13, h21, h22, h23, h31, h32] = params;
        const [x, y] = point;
        const denom = (h31 * x) + (h32 * y) + 1;
        if (Math.abs(denom) < 1e-9) return null;
        return [((h11 * x) + (h12 * y) + h13) / denom, ((h21 * x) + (h22 * y) + h23) / denom];
      };
      const features = [];
      (panel.reference_route_geojson.features || []).forEach((feature) => {
        const geometry = feature.geometry || {};
        const lines = geometry.type === 'LineString' ? [geometry.coordinates || []] : (geometry.coordinates || []);
        lines.forEach((line, index) => {
          const coords = line
            .map((coord) => project(pagePtToImagePx(panel, coord)))
            .filter(Boolean)
            .map((coord) => [Number(coord[0].toFixed(7)), Number(coord[1].toFixed(7))]);
          if (coords.length >= 2) {
            features.push({
              type: 'Feature',
              geometry: { type: 'LineString', coordinates: coords },
              properties: { ...feature.properties, source_line_index: index },
            });
          }
        });
      });
      return { type: 'FeatureCollection', features };
    }

    function buildTempleGeojsonForMap(panel) {
      return {
        type: 'FeatureCollection',
        features: templePoints(panel).map((point) => ({
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: [Number(point.longitude), Number(point.latitude)],
          },
          properties: {
            gcp_id: point.gcp_id || '',
            temple_group: point.temple_group || '',
            temple_no: point.temple_no || '',
            temple_no_label: point.temple_no_label || '',
            name_full: point.name_full || '',
            name_short: point.name_short || '',
            source_name_text: point.source_name_text || '',
            confidence: point.confidence ?? '',
            source_kind: point.source_kind || '',
            review_reasons: point.review_reasons || '',
          },
        })),
      };
    }

    function setLayerVisibility(layerIds, visible) {
      layerIds.forEach((layerId) => {
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
        }
      });
    }

    function updateMapSources() {
      if (!mapReady) return;
      const panel = panelById(currentPanelId);
      ensureMapSource('all-saved-routes', buildAllManualRoutesSourceData());
      ensureMapSource('current-panel-routes', buildCurrentPanelRoutesSourceData(currentPanelId));
      ensureMapSource('active-route', lineFeatureForActiveRoute(activeRoute()));
      ensureMapSource('reference-routes', buildReferenceGeojsonForMap(panel));
      ensureMapSource('temple-points', buildTempleGeojsonForMap(panel));
      setLayerVisibility(['all-saved-routes-line'], allRoutesVisible);
      setLayerVisibility(['reference-routes-line'], referenceVisible);
      setLayerVisibility(['temple-circle', 'temple-number', 'temple-label'], templeVisible);
    }

    function clearVertexMarkers() {
      vertexMarkers.forEach((marker) => marker.remove());
      vertexMarkers = [];
    }

    function refreshVertexMarkers() {
      clearVertexMarkers();
      if (!mapReady) return;
      const route = activeRoute();
      if (!route) return;
      route.coordinates.forEach((coord, index) => {
        const el = document.createElement('div');
        el.style.width = '14px';
        el.style.height = '14px';
        el.style.borderRadius = '999px';
        el.style.border = '2px solid white';
        el.style.background = '#ef4444';
        el.style.boxShadow = '0 0 0 1px rgba(15,23,42,0.24)';
        if (index === route.coordinates.length - 1) el.style.background = '#2563eb';
        const marker = new mapboxgl.Marker({ element: el, draggable: currentEditMode() === 'edit' })
          .setLngLat(coord)
          .addTo(map);
        marker.getElement().addEventListener('click', (event) => {
          event.stopPropagation();
          if (currentEditMode() === 'delete') {
            pushHistory(currentPanelId);
            route.coordinates.splice(index, 1);
            route.updated_at = new Date().toISOString();
            if (route.coordinates.length < 2) {
              removeRoute(currentPanelId, route.route_id);
              return;
            }
            touchPanel(currentPanelId);
            renderAll();
          } else {
            refreshVertexMarkers();
          }
        });
        marker.on('dragend', () => {
          pushHistory(currentPanelId);
          const lngLat = marker.getLngLat();
          route.coordinates[index] = [Number(lngLat.lng.toFixed(7)), Number(lngLat.lat.toFixed(7))];
          route.updated_at = new Date().toISOString();
          touchPanel(currentPanelId);
          renderAll();
        });
        vertexMarkers.push(marker);
      });
    }

    function fitMapToPointRows(rows, zoomForSingle = 14.5) {
      if (!rows.length) return false;
      if (rows.length === 1) {
        map.easeTo({
          center: [Number(rows[0].longitude), Number(rows[0].latitude)],
          zoom: zoomForSingle,
          duration: 0,
        });
        return true;
      }
      const bounds = new mapboxgl.LngLatBounds();
      rows.forEach((row) => bounds.extend([Number(row.longitude), Number(row.latitude)]));
      map.fitBounds(bounds, { padding: 60, duration: 0, maxZoom: 15.8 });
      return true;
    }

    function rememberCurrentMapView() {
      if (!mapReady) return;
      const center = map.getCenter();
      state.lastMapView = {
        center: [Number(center.lng.toFixed(7)), Number(center.lat.toFixed(7))],
        zoom: Number(map.getZoom().toFixed(3)),
      };
      localStorage.setItem(config.local_storage_key, JSON.stringify(buildAutosavePayload()));
    }

    function fitMapToPanel(panel) {
      if (!mapReady) return;
      if (!panel) return;
      const routeFeatures = buildCurrentPanelRoutesSourceData(panel.georef_panel_id).features;
      if (routeFeatures.length) {
        const bounds = new mapboxgl.LngLatBounds();
        routeFeatures.forEach((feature) => feature.geometry.coordinates.forEach((coord) => bounds.extend(coord)));
        map.fitBounds(bounds, { padding: 50, duration: 0, maxZoom: 16.5 });
        return;
      }
      if (panel.saved_corners) {
        const bounds = new mapboxgl.LngLatBounds();
        ['top_left', 'top_right', 'bottom_right', 'bottom_left'].forEach((key) => bounds.extend(panel.saved_corners[key]));
        map.fitBounds(bounds, { padding: 50, duration: 0, maxZoom: 16.5 });
        return;
      }
      if (fitMapToPointRows(manualGeoPoints(panel), 14.8)) return;
      if (fitMapToPointRows(autoGeoPoints(panel), 14.4)) return;
      if (fitMapToPointRows(templePoints(panel), 14.2)) return;
      if (state.lastMapView?.center?.length === 2) {
        map.jumpTo({ center: state.lastMapView.center, zoom: state.lastMapView.zoom || 12 });
        return;
      }
      map.jumpTo({ center: panel.initial_center || config.initial_map_center || DEFAULT_CENTER, zoom: panel.initial_zoom || config.initial_map_zoom || DEFAULT_ZOOM });
    }

    function nearestSegmentInsertIndex(route, lngLat) {
      if (!route || route.coordinates.length < 2 || !mapReady) return null;
      const clickPoint = map.project(lngLat);
      let best = null;
      for (let index = 0; index < route.coordinates.length - 1; index += 1) {
        const a = map.project(route.coordinates[index]);
        const b = map.project(route.coordinates[index + 1]);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const lenSq = (dx * dx) + (dy * dy);
        if (lenSq < 1e-6) continue;
        const t = Math.max(0, Math.min(1, (((clickPoint.x - a.x) * dx) + ((clickPoint.y - a.y) * dy)) / lenSq));
        const px = a.x + (dx * t);
        const py = a.y + (dy * t);
        const dist = Math.hypot(clickPoint.x - px, clickPoint.y - py);
        if (!best || dist < best.distance) best = { index, distance: dist };
      }
      return best && best.distance <= 12 ? best.index + 1 : null;
    }

    function onMapClick(event) {
      if (map.queryRenderedFeatures(event.point, { layers: ['temple-circle', 'temple-number', 'temple-label'] }).length) {
        return;
      }
      const panel = panelById(currentPanelId);
      const coord = [Number(event.lngLat.lng.toFixed(7)), Number(event.lngLat.lat.toFixed(7))];
      let route = activeRoute();
      if (event.originalEvent.shiftKey || !route) {
        route = startNewRoute(currentPanelId, coord);
        renderAll();
        return;
      }
      pushHistory(currentPanelId);
      const insertIndex = nearestSegmentInsertIndex(route, event.lngLat);
      if (insertIndex !== null) route.coordinates.splice(insertIndex, 0, coord);
      else route.coordinates.push(coord);
      route.updated_at = new Date().toISOString();
      touchPanel(currentPanelId);
      renderAll();
    }

    function onMapDoubleClick(event) {
      event.preventDefault();
      finalizeActiveRoute();
    }

    function onMapRouteClick(event) {
      const feature = event.features?.[0];
      if (!feature) return;
      setActiveRouteId(currentPanelId, feature.properties.route_id);
      renderAll();
    }

    function onAllRoutesClick(event) {
      const feature = event.features?.[0];
      if (!feature) return;
      const props = feature.properties || {};
      const coordinates = feature.geometry?.coordinates || [];
      const anchor = coordinates[Math.floor(coordinates.length / 2)] || coordinates[0];
      if (!anchor) return;
      new mapboxgl.Popup({ closeButton: true, offset: 10 })
        .setLngLat(anchor)
        .setHTML(
          `<strong>p${props.page_no} / ${props.georef_panel_id || '-'}</strong><br>` +
          `length_km=${((Number(props.length_m || 0)) / 1000).toFixed(2)}<br>` +
          `vertex=${props.vertex_count || '-'} / status=${props.status || '-'}`
        )
        .addTo(map);
    }

    function onTempleClick(event) {
      const feature = event.features?.[0];
      if (!feature) return;
      const props = feature.properties || {};
      const coordinates = feature.geometry?.coordinates || [0, 0];
      new mapboxgl.Popup({ closeButton: true, offset: 14 })
        .setLngLat(coordinates)
        .setHTML(
          `<strong>${props.temple_no_label || '-'} ${props.name_short || ''}</strong><br>` +
          `${props.name_full || ''}<br>` +
          `group=${props.temple_group || '-'} / source=${props.source_kind || '-'}<br>` +
          `lon=${Number(coordinates[0]).toFixed(6)}, lat=${Number(coordinates[1]).toFixed(6)}`
        )
        .addTo(map);
    }

    function updateRouteList() {
      const routes = panelRoutes(currentPanelId);
      if (!routes.length) {
        ui.routeList.innerHTML = '<div class="routeMeta">まだLineStringがありません。</div>';
        return;
      }
      ui.routeList.innerHTML = routes.map((route) => {
        const active = route.route_id === activeRouteId();
        const km = (routeLengthMeters(route.coordinates) / 1000).toFixed(2);
        return `
          <div class="routeRow ${active ? 'active' : ''}" data-route-id="${route.route_id}">
            <div>
              <div><strong>${route.route_id}</strong></div>
              <div class="routeMeta">status=${route.status} / vertex=${route.coordinates.length} / ${km} km</div>
            </div>
            <button class="routeDelete" data-delete-route-id="${route.route_id}">削除</button>
          </div>
        `;
      }).join('');
    }

    function renderMapDebug() {
      const panel = panelById(currentPanelId);
      const summary = computePanelSummary(currentPanelId);
      const route = activeRoute();
      ui.mapDebug.textContent = [
        `page_no: ${panel.page_no}`,
        `georef_panel_id: ${panel.georef_panel_id}`,
        `panel_type: ${panel.panel_type}`,
        `status: ${summary.status}`,
        `route_count: ${panelRoutes(currentPanelId).length}`,
        `vertex_count: ${summary.vertexCount}`,
        `panel_length_km: ${(summary.lengthKm).toFixed(3)}`,
        `active_route: ${route ? route.route_id : '-'}`,
        `all_routes_visible: ${allRoutesVisible}`,
        `reference_on_map: ${referenceVisible}`,
        `temple_markers: ${templeVisible} (${templePoints(panel).length})`,
      ].join('\\n');
    }

    function renderAll() {
      populatePanelSelect();
      updateStepText();
      updateRouteList();
      ui.panelNotes.value = panelNotes(currentPanelId);
      drawPdfPane();
      updateMapSources();
      refreshVertexMarkers();
      renderMapDebug();
    }

    async function writeFileToHandle(handle, filename, contents) {
      const fileHandle = await handle.getFileHandle(filename, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(contents);
      await writable.close();
    }

    async function writeRouteSegments(handle) {
      const dirHandle = await handle.getDirectoryHandle('route_segments', { create: true });
      for (const panel of sortedPanels()) {
        const features = buildCurrentPanelRoutesSourceData(panel.georef_panel_id).features;
        const payload = JSON.stringify({ type: 'FeatureCollection', features }, null, 2);
        await writeFileToHandle(dirHandle, `${panel.georef_panel_id}.geojson`, payload);
      }
    }

    async function persistToOutputDir() {
      if (!outputDirHandle) return false;
      const manualRoutes = JSON.stringify(buildManualRoutesGeojson(), null, 2);
      const autosave = JSON.stringify(buildAutosavePayload(), null, 2);
      const report = [
        '# Manual Route Editor Report',
        '',
        `generated_at: ${new Date().toISOString()}`,
        `panel_count: ${config.panels.length}`,
        `route_feature_count: ${buildManualRoutesGeojson().features.length}`,
        '',
        'This file is maintained by the manual route editor.',
      ].join('\\n');
      await writeFileToHandle(outputDirHandle, 'manual_routes.geojson', manualRoutes);
      await writeFileToHandle(outputDirHandle, 'autosave.json', autosave);
      await writeRouteSegments(outputDirHandle);
      await writeFileToHandle(outputDirHandle, 'report.md', report);
      return true;
    }

    function downloadText(filename, text, mimeType = 'application/json') {
      const blob = new Blob([text], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    }

    let autosaveTimer = null;
    function scheduleAutosave() {
      localStorage.setItem(config.local_storage_key, JSON.stringify(buildAutosavePayload()));
      clearTimeout(autosaveTimer);
      autosaveTimer = setTimeout(async () => {
        try {
          await persistToOutputDir();
        } catch (error) {
          ui.notice.textContent = `autosave warning: ${error.message}`;
        }
      }, 150);
    }

    async function onSave() {
      try {
        localStorage.setItem(config.local_storage_key, JSON.stringify(buildAutosavePayload()));
        const persisted = await persistToOutputDir();
        if (!persisted) {
          downloadText('manual_routes.geojson', JSON.stringify(buildManualRoutesGeojson(), null, 2));
          downloadText('autosave.json', JSON.stringify(buildAutosavePayload(), null, 2));
        }
        ui.notice.textContent = persisted
          ? 'manual_routes.geojson / autosave.json / route_segments/ に保存しました'
          : 'dir未バインドのため JSON をダウンロードしました';
        renderMapDebug();
      } catch (error) {
        ui.notice.textContent = `save error: ${error.message}`;
      }
    }

    async function bindOutputDir() {
      if (!window.showDirectoryPicker) {
        ui.notice.textContent = 'このブラウザは File System Access API をサポートしていません';
        return;
      }
      try {
        outputDirHandle = await window.showDirectoryPicker({ id: 'manual-route-editor-output' });
        ui.notice.textContent = 'output dir をバインドしました';
        await persistToOutputDir();
      } catch (error) {
        if (error && error.name === 'AbortError') return;
        ui.notice.textContent = `bind error: ${error.message}`;
      }
    }

    async function loadPersistedState() {
      const stored = localStorage.getItem(config.local_storage_key);
      let localPayload = null;
      if (stored) {
        try { localPayload = JSON.parse(stored); } catch {}
      }
      let autosavePayload = null;
      try {
        const response = await fetch('autosave.json', { cache: 'no-store' });
        if (response.ok) autosavePayload = await response.json();
      } catch {}
      let routesPayload = null;
      try {
        const response = await fetch('manual_routes.geojson', { cache: 'no-store' });
        if (response.ok) routesPayload = await response.json();
      } catch {}

      const preferred = localPayload || autosavePayload;
      if (preferred?.panels) {
        state.panels = preferred.panels;
        state.activeRouteIdByPanel = preferred.active_route_id_by_panel || {};
      }
      if (preferred?.last_map_view?.center?.length === 2) {
        state.lastMapView = preferred.last_map_view;
      }
      if (routesPayload?.features?.length && !preferred?.panels) {
        routesPayload.features.forEach((feature) => {
          const props = feature.properties || {};
          const panelId = props.georef_panel_id;
          if (!panelId) return;
          const panelState = ensurePanelState(panelId);
          panelState.routes.push({
            route_id: props.route_id,
            coordinates: feature.geometry.coordinates || [],
            status: props.status || 'draft',
            created_at: props.created_at || new Date().toISOString(),
            updated_at: props.updated_at || props.created_at || new Date().toISOString(),
            notes: props.notes || '',
          });
        });
      }
      sortedPanels().forEach((panel) => ensurePanelState(panel.georef_panel_id));
    }

    function onPanelChange() {
      currentPanelId = ui.panelSelect.value;
      const panel = panelById(currentPanelId);
      const view = getPanelViewState(panel);
      view.fitMode = 'all';
      view.fitApplied = false;
      renderAll();
      fitMapToPanel(panel);
    }

    function stepPanel(offset) {
      const panels = sortedPanels();
      const index = panels.findIndex((panel) => panel.georef_panel_id === currentPanelId);
      const next = panels[(index + offset + panels.length) % panels.length];
      currentPanelId = next.georef_panel_id;
      ui.panelSelect.value = currentPanelId;
      const panel = panelById(currentPanelId);
      const view = getPanelViewState(panel);
      view.fitMode = 'all';
      view.fitApplied = false;
      renderAll();
      fitMapToPanel(panel);
    }

    function onNotesInput() {
      ensurePanelState(currentPanelId).notes = ui.panelNotes.value;
      touchPanel(currentPanelId);
    }

    function onUndo() {
      undoPanel(currentPanelId);
    }

    function removeLastVertex() {
      const route = activeRoute();
      if (!route) return;
      pushHistory(currentPanelId);
      route.coordinates.pop();
      route.updated_at = new Date().toISOString();
      if (!route.coordinates.length) {
        removeRoute(currentPanelId, route.route_id);
        return;
      }
      touchPanel(currentPanelId);
      renderAll();
    }

    function setReferenceVisible(nextVisible) {
      referenceVisible = nextVisible;
      ui.toggleReferenceButton.textContent = referenceVisible ? 'Reference ON' : 'Reference OFF';
      updateMapSources();
      renderMapDebug();
    }

    function setAllRoutesVisible(nextVisible) {
      allRoutesVisible = nextVisible;
      ui.toggleAllRoutesButton.textContent = allRoutesVisible ? 'All Routes ON' : 'All Routes OFF';
      updateMapSources();
      renderMapDebug();
    }

    function setTempleVisible(nextVisible) {
      templeVisible = nextVisible;
      ui.toggleTempleButton.textContent = templeVisible ? 'Temple ON' : 'Temple OFF';
      updateMapSources();
      renderMapDebug();
    }

    function initMap() {
      if (!config.mapbox_access_token) {
        ui.notice.textContent = 'MAPBOX_ACCESS_TOKEN がありません';
        return;
      }
      mapboxgl.accessToken = config.mapbox_access_token;
      map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/outdoors-v12',
        center: config.initial_map_center || DEFAULT_CENTER,
        zoom: config.initial_map_zoom || DEFAULT_ZOOM,
      });
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');
      map.on('load', () => {
        mapReady = true;
        ensureMapSource('all-saved-routes', { type: 'FeatureCollection', features: [] });
        ensureMapSource('current-panel-routes', { type: 'FeatureCollection', features: [] });
        ensureMapSource('active-route', { type: 'FeatureCollection', features: [] });
        ensureMapSource('reference-routes', { type: 'FeatureCollection', features: [] });
        ensureMapSource('temple-points', { type: 'FeatureCollection', features: [] });
        map.addLayer({
          id: 'reference-routes-line',
          type: 'line',
          source: 'reference-routes',
          layout: { visibility: 'none' },
          paint: {
            'line-color': '#9ca3af',
            'line-width': 2,
            'line-opacity': 0.65,
          },
        });
        map.addLayer({
          id: 'temple-circle',
          type: 'circle',
          source: 'temple-points',
          paint: {
            'circle-radius': 8,
            'circle-color': '#1d4ed8',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
          },
        });
        map.addLayer({
          id: 'temple-number',
          type: 'symbol',
          source: 'temple-points',
          layout: {
            'text-field': ['coalesce', ['get', 'temple_no_label'], ''],
            'text-size': 10,
            'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
            'text-offset': [0, 0],
          },
          paint: {
            'text-color': '#ffffff',
          },
        });
        map.addLayer({
          id: 'temple-label',
          type: 'symbol',
          source: 'temple-points',
          layout: {
            'text-field': ['coalesce', ['get', 'name_short'], ''],
            'text-size': 11,
            'text-font': ['Open Sans Semibold', 'Arial Unicode MS Regular'],
            'text-offset': [0, 1.45],
            'text-anchor': 'top',
          },
          paint: {
            'text-color': '#0f172a',
            'text-halo-color': '#ffffff',
            'text-halo-width': 1.2,
          },
        });
        map.addLayer({
          id: 'all-saved-routes-line',
          type: 'line',
          source: 'all-saved-routes',
          paint: {
            'line-color': '#991b1b',
            'line-width': 2,
            'line-opacity': 0.35,
          },
        });
        map.addLayer({
          id: 'current-panel-routes-line',
          type: 'line',
          source: 'current-panel-routes',
          paint: {
            'line-color': '#dc2626',
            'line-width': 4,
            'line-opacity': 0.9,
          },
        });
        map.addLayer({
          id: 'active-route-line',
          type: 'line',
          source: 'active-route',
          paint: {
            'line-color': '#dc2626',
            'line-width': 5,
            'line-opacity': 0.95,
          },
        });
        map.on('click', onMapClick);
        map.on('dblclick', onMapDoubleClick);
        map.on('click', 'current-panel-routes-line', onMapRouteClick);
        map.on('click', 'all-saved-routes-line', onAllRoutesClick);
        map.on('click', 'temple-circle', onTempleClick);
        map.on('click', 'temple-number', onTempleClick);
        map.on('click', 'temple-label', onTempleClick);
        map.on('moveend', rememberCurrentMapView);
        fitMapToPanel(panelById(currentPanelId));
        renderAll();
      });
    }

    function wireUi() {
      ui.panelSelect.addEventListener('change', onPanelChange);
      ui.prevButton.addEventListener('click', () => stepPanel(-1));
      ui.nextButton.addEventListener('click', () => stepPanel(1));
      ui.bindDirButton.addEventListener('click', bindOutputDir);
      ui.newRouteButton.addEventListener('click', () => { startNewRoute(currentPanelId); renderAll(); });
      ui.undoButton.addEventListener('click', onUndo);
      ui.saveButton.addEventListener('click', onSave);
      ui.toggleAllRoutesButton.addEventListener('click', () => setAllRoutesVisible(!allRoutesVisible));
      ui.toggleReferenceButton.addEventListener('click', () => setReferenceVisible(!referenceVisible));
      ui.toggleTempleButton.addEventListener('click', () => setTempleVisible(!templeVisible));
      ui.panelNotes.addEventListener('input', onNotesInput);
      ui.routeList.addEventListener('click', (event) => {
        const deleteButton = event.target.closest('[data-delete-route-id]');
        if (deleteButton) {
          removeRoute(currentPanelId, deleteButton.dataset.deleteRouteId);
          return;
        }
        const row = event.target.closest('[data-route-id]');
        if (!row) return;
        setActiveRouteId(currentPanelId, row.dataset.routeId);
        renderAll();
      });
      ui.pdfDisplaySelect.addEventListener('change', renderAll);
      ui.editModeSelect.addEventListener('change', refreshVertexMarkers);
      ui.pdfFitAllButton.addEventListener('click', () => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        const rect = ui.pdfCanvas.getBoundingClientRect();
        applyPdfFit(panel, rect, view, 'all');
        drawPdfPane();
      });
      ui.pdfFitWidthButton.addEventListener('click', () => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        const rect = ui.pdfCanvas.getBoundingClientRect();
        applyPdfFit(panel, rect, view, 'width');
        drawPdfPane();
      });
      ui.pdfFitHeightButton.addEventListener('click', () => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        const rect = ui.pdfCanvas.getBoundingClientRect();
        applyPdfFit(panel, rect, view, 'height');
        drawPdfPane();
      });
      ui.pdfZoomInButton.addEventListener('click', () => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        view.zoom = Math.min(8, view.zoom * 1.12);
        drawPdfPane();
      });
      ui.pdfZoomOutButton.addEventListener('click', () => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        view.zoom = Math.max(0.5, view.zoom / 1.12);
        drawPdfPane();
      });
      ui.pdfCanvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        view.dragState = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          panX: view.panX,
          panY: view.panY,
        };
        ui.pdfCanvas.classList.add('dragging');
        ui.pdfCanvas.setPointerCapture(event.pointerId);
      });
      ui.pdfCanvas.addEventListener('pointermove', (event) => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        if (!view.dragState || view.dragState.pointerId !== event.pointerId) return;
        view.panX = view.dragState.panX + (event.clientX - view.dragState.startX);
        view.panY = view.dragState.panY + (event.clientY - view.dragState.startY);
        drawPdfPane();
      });
      const stopPdfDrag = (event) => {
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        if (!view.dragState) return;
        if (event.pointerId !== undefined && view.dragState.pointerId !== event.pointerId) return;
        view.dragState = null;
        ui.pdfCanvas.classList.remove('dragging');
      };
      ui.pdfCanvas.addEventListener('pointerup', stopPdfDrag);
      ui.pdfCanvas.addEventListener('pointercancel', stopPdfDrag);
      ui.pdfCanvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const panel = panelById(currentPanelId);
        const view = getPanelViewState(panel);
        const rect = ui.pdfCanvas.getBoundingClientRect();
        const pointerX = event.clientX - rect.left;
        const pointerY = event.clientY - rect.top;
        const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
        zoomPdfAtPoint(panel, view, rect, pointerX, pointerY, factor);
        drawPdfPane();
      }, { passive: false });
      window.addEventListener('keydown', (event) => {
        if (event.target && ['TEXTAREA', 'INPUT', 'SELECT'].includes(event.target.tagName)) return;
        if (event.key === 'Backspace') {
          event.preventDefault();
          removeLastVertex();
        } else if (event.key.toLowerCase() === 'z') {
          event.preventDefault();
          onUndo();
        } else if (event.key.toLowerCase() === 's') {
          event.preventDefault();
          onSave();
        } else if (event.key.toLowerCase() === 'n') {
          event.preventDefault();
          stepPanel(1);
        } else if (event.key.toLowerCase() === 'p') {
          event.preventDefault();
          stepPanel(-1);
        } else if (event.key.toLowerCase() === 'e') {
          ui.editModeSelect.value = 'edit';
          refreshVertexMarkers();
        } else if (event.key.toLowerCase() === 'd') {
          ui.editModeSelect.value = 'delete';
          refreshVertexMarkers();
        } else if (event.key === 'Escape') {
          setActiveRouteId(currentPanelId, null);
          renderAll();
        }
      });
      window.addEventListener('resize', () => {
        const panel = panelById(currentPanelId);
        if (!panel) return;
        getPanelViewState(panel).fitApplied = false;
        drawPdfPane();
      });
    }

    async function init() {
      await loadPersistedState();
      const panels = sortedPanels();
      currentPanelId = config.default_georef_panel_id || panels[0]?.georef_panel_id || '';
      populatePanelSelect();
      wireUi();
      await drawPdfPane();
      updateStepText();
      updateRouteList();
      renderMapDebug();
      initMap();
    }

    init();
  </script>
</body>
</html>
"""
    return template.replace("__CONFIG_JSON__", json.dumps(config, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual route editor artifacts.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--panels-csv", type=Path)
    parser.add_argument("--panels-geojson", type=Path)
    parser.add_argument("--accepted-main-routes", type=Path)
    parser.add_argument("--accepted-inset-routes", type=Path)
    parser.add_argument("--gazetteer-csv", type=Path)
    parser.add_argument("--gcp-candidates-csv", type=Path)
    parser.add_argument("--saved-georef-dir", type=Path, default=Path("data/manual_image_georef"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/manual_route_editor"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--georef-panel-id", action="append", default=[], help="Process only specific georef_panel_id. Repeatable.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render-scale", type=float, default=RENDER_SCALE)
    args = parser.parse_args()

    image_repo_dir = takram_image_dir(args.env_file)
    panel_route_dir = image_repo_dir / "artifacts" / "panel_route_detection"
    external_step4_dir = image_repo_dir / "artifacts" / "step4"
    local_step4_dir = args.env_file.parent / "artifacts" / "step4"
    panels_csv = args.panels_csv or (panel_route_dir / "georef_panels.csv")
    panels_geojson = args.panels_geojson or (panel_route_dir / "georef_panels.geojson")
    accepted_main_routes = args.accepted_main_routes or (panel_route_dir / "accepted_main_georef_routes.geojson")
    accepted_inset_routes = args.accepted_inset_routes or (panel_route_dir / "accepted_inset_georef_routes.geojson")
    gazetteer_csv = args.gazetteer_csv or first_existing_path([external_step4_dir / "gazetteer.csv", local_step4_dir / "gazetteer.csv"])
    gcp_candidates_csv = args.gcp_candidates_csv or first_existing_path([external_step4_dir / "gcp_candidates.csv", local_step4_dir / "gcp_candidates.csv"])

    ensure_dir(args.out_dir)
    image_dir = args.out_dir / "images"
    ensure_dir(image_dir)
    ensure_dir(args.out_dir / "route_segments")

    panels = [
        row for row in read_csv_rows(panels_csv)
        if str(row.get("keep_for_georef", "")).strip().lower() == "true"
    ]
    panel_features = panel_geojson_by_key(panels_geojson)
    reference_routes_by_panel = route_features_by_panel([accepted_main_routes, accepted_inset_routes])
    gcp_rows = read_optional_csv_rows(gcp_candidates_csv)
    gcp_rows_for_panel = gcp_rows_by_panel(panels, gcp_rows)
    saved_georef = saved_georef_by_panel(args.saved_georef_dir)
    token = read_mapbox_token(args.env_file)

    selected_panel_ids = set(args.georef_panel_id)
    panel_rows: list[dict[str, Any]] = []
    for panel_row in panels:
        panel_id = str(panel_row["georef_panel_id"])
        if selected_panel_ids and panel_id not in selected_panel_ids:
            continue
        page_no = int(panel_row["page_no"])
        panel_row["__route_count"] = len(reference_routes_by_panel.get((page_no, panel_id), []))
        panel_rows.append(panel_row)

    panel_rows.sort(key=lambda row: (int(row["page_no"]), row["georef_panel_id"]))
    if args.limit > 0:
        panel_rows = panel_rows[:args.limit]

    manifest_panels: list[dict[str, Any]] = []
    doc = fitz.open(args.pdf)
    try:
        for panel_row in panel_rows:
            page_no = int(panel_row["page_no"])
            panel_id = panel_row["georef_panel_id"]
            key = (page_no, panel_id)
            map_image, redlines_image = render_panel_images(
                doc=doc,
                page_no=page_no,
                panel_row=panel_row,
                render_scale=args.render_scale,
            )
            map_path = image_dir / f"{panel_id}_map.png"
            redlines_path = image_dir / f"{panel_id}_redlines.png"
            map_image.save(map_path)
            redlines_image.save(redlines_path)

            saved_row = saved_georef.get(key)
            saved_corners = saved_row.get("corners_lonlat") if saved_row else None
            initial_center, initial_zoom = choose_initial_view(saved_row, map_image.width, map_image.height)
            panel_bbox = rounded_bbox(panel_row)
            panel_area_pt2 = bbox_area_pt2(panel_bbox)
            panel_feature = panel_features.get(key)
            reference_features = reference_routes_by_panel.get(key, [])
            panel_gcp_rows = gcp_rows_for_panel.get(key, [])
            manifest_panels.append(
                {
                    "page_no": page_no,
                    "georef_panel_id": panel_id,
                    "panel_type": panel_row.get("panel_type", ""),
                    "layout_panel_id": panel_row.get("layout_panel_id", ""),
                    "source_panel_type": panel_row.get("source_panel_type", ""),
                    "route_count": len(reference_features),
                    "image_path": f"images/{panel_id}_map.png",
                    "redlines_path": f"images/{panel_id}_redlines.png",
                    "reference_route_geojson": {"type": "FeatureCollection", "features": reference_features},
                    "pdf_bbox": panel_bbox,
                    "panel_polygon_pdf": polygon_coords_from_feature(panel_feature),
                    "image_width_px": map_image.width,
                    "image_height_px": map_image.height,
                    "initial_center": initial_center,
                    "initial_zoom": initial_zoom,
                    "saved_corners": saved_corners,
                    "saved_georef_path": str(saved_row.get("__path__", "")) if saved_row else "",
                    "manual_geo_points": serializable_manual_geo_points(saved_row),
                    "auto_geo_points": serializable_auto_geo_points(panel_gcp_rows),
                    "temple_points": serializable_temple_points(panel_gcp_rows),
                    "panel_area_pt2": round(panel_area_pt2, 3),
                }
            )
    finally:
        doc.close()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(args.pdf),
        "mapbox_access_token": token,
        "initial_map_center": DEFAULT_CENTER,
        "initial_map_zoom": DEFAULT_ZOOM,
        "default_georef_panel_id": manifest_panels[0]["georef_panel_id"] if manifest_panels else "",
        "panel_count": len(manifest_panels),
        "panels": manifest_panels,
        "editor_version": EDITOR_VERSION,
        "local_storage_key": "manual-route-editor-autosave",
        "gazetteer_csv": str(gazetteer_csv) if gazetteer_csv else "",
        "gcp_candidates_csv": str(gcp_candidates_csv) if gcp_candidates_csv else "",
    }

    (args.out_dir / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "index.html").write_text(build_html(manifest), encoding="utf-8")
    (args.out_dir / "manual_routes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "autosave.json").write_text(json.dumps({"editor_version": EDITOR_VERSION, "panels": {}, "active_route_id_by_panel": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Manual Route Editor",
                "",
                "左に PDF georef_panel crop、右に Mapbox を表示し、人間が Mapbox 上で直接ルートを手打ちするためのエディタです。",
                "",
                "## 使い方",
                "",
                "1. `python -m http.server 8131 -d artifacts/manual_route_editor` で配信します。",
                "2. 最初に `Bind Output Dir` で `artifacts/manual_route_editor` を選びます。",
                "3. 左は PDF 参照、右は Mapbox 手打ちです。寺マーカーを目印にします。",
                "4. 右 Mapbox をクリックして頂点を追加し、ダブルクリックで現在の LineString を確定します。",
                "5. `S` で保存、`Z` で Undo、`N/P` で panel 移動します。",
                "",
                "Accepted routes reference は参考表示であり、最終採用しません。",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(manifest_panels)} panels to {args.out_dir}")
    print(f"  default_panel={manifest['default_georef_panel_id'] or '(empty)'}")


if __name__ == "__main__":
    main()
