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
import numpy as np
from PIL import Image


MAPBOX_SCALE = 0.01
EARTH_RADIUS_M = 6378137.0
DEFAULT_CENTER = [133.5, 33.8]
DEFAULT_ZOOM = 8.5
RENDER_SCALE = 2.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_mapbox_token(env_path: Path) -> str:
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MAPBOX_ACCESS_TOKEN="):
            return line.partition("=")[2].strip()
    return ""


def route_features_by_frame(path: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        page_no = props.get("page_no")
        frame_id = props.get("frame_id")
        if page_no in (None, "") or not frame_id:
            continue
        grouped[(int(page_no), str(frame_id))].append(feature)
    return grouped


def priority_by_frame(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = read_csv_rows(path)
    return {
        (int(row["page_no"]), row["frame_id"]): row
        for row in rows
        if row.get("page_no") and row.get("frame_id")
    }


def frame_models_by_frame(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    return {
        (int(row["page_no"]), row["frame_id"]): row
        for row in rows
        if row.get("page_no") and row.get("frame_id")
    }


def gcps_by_frame(path: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    if not path.exists():
        return {}
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv_rows(path):
        if not row.get("page_no") or not row.get("frame_id"):
            continue
        grouped[(int(row["page_no"]), row["frame_id"])].append(row)
    return grouped


def manual_gcps_by_frame(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return result
    for json_path in sorted(path.glob("page_*_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        page_no = data.get("page_no")
        frame_id = data.get("frame_id")
        if page_no in (None, "") or not frame_id:
            continue
        result[(int(page_no), str(frame_id))] = data
    return result


def saved_georef_by_frame(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return result
    for json_path in sorted(path.glob("page_*_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        page_no = data.get("page_no")
        frame_id = data.get("frame_id")
        if page_no in (None, "") or not frame_id:
            continue
        data["__path__"] = str(json_path)
        result[(int(page_no), str(frame_id))] = data
    return result


def rounded_bbox(frame_row: dict[str, Any]) -> dict[str, float]:
    return {
        "x0": round(float(frame_row["x0_pt"]), 3),
        "y0": round(float(frame_row["y0_pt"]), 3),
        "x1": round(float(frame_row["x1_pt"]), 3),
        "y1": round(float(frame_row["y1_pt"]), 3),
    }


def render_frame_images(
    *,
    doc: fitz.Document,
    page_no: int,
    frame_row: dict[str, Any],
    render_scale: float,
) -> tuple[Image.Image, Image.Image]:
    page = doc[page_no - 1]
    clip = fitz.Rect(
        float(frame_row["x0_pt"]),
        float(frame_row["y0_pt"]),
        float(frame_row["x1_pt"]),
        float(frame_row["y1_pt"]),
    )
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


def save_feature_collection(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def geo_points_from_manual_data(data: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in data.get("gcps", []):
        if row.get("role") != "frame_anchor":
            continue
        if row.get("longitude") in (None, "") or row.get("latitude") in (None, ""):
            continue
        points.append((float(row["longitude"]), float(row["latitude"])))
    return points


def geo_points_from_auto_gcps(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row.get("longitude") in (None, "") or row.get("latitude") in (None, ""):
            continue
        points.append((float(row["longitude"]), float(row["latitude"])))
    return points


def lonlat_to_local_equirect(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return (x, y)


def local_equirect_to_lonlat(x: float, y: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    return (lon, lat)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    covariance = (tgt_centered.T @ src_centered) / len(source)
    u, d, vt = np.linalg.svd(covariance)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rotation = u @ s @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if variance <= 1e-12:
        raise np.linalg.LinAlgError("degenerate source variance")
    scale = np.trace(np.diag(d) @ s) / variance
    translation = tgt_mean - (scale * (rotation @ src_mean))
    return {"rotation": rotation, "scale": scale, "translation": translation}


def apply_similarity(points: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    rotation = params["rotation"]
    scale = float(params["scale"])
    translation = params["translation"]
    return ((scale * (rotation @ points.T)).T) + translation


def fit_projective(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    matrix = np.array(rows, dtype=float)
    rhs = np.array(values, dtype=float)
    params, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return params


def apply_projective(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    h11, h12, h13, h21, h22, h23, h31, h32 = params
    x = points[:, 0]
    y = points[:, 1]
    denom = (h31 * x) + (h32 * y) + 1.0
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    u = ((h11 * x) + (h12 * y) + h13) / denom
    v = ((h21 * x) + (h22 * y) + h23) / denom
    return np.column_stack([u, v])


def manual_anchor_rows(data: dict[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in data.get("gcps", []):
        if row.get("role") != "frame_anchor":
            continue
        if any(row.get(key) in (None, "") for key in ("pdf_x", "pdf_y", "longitude", "latitude")):
            continue
        rows.append(
            {
                "pdf_x": float(row["pdf_x"]),
                "pdf_y": float(row["pdf_y"]),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
            }
        )
    return rows


def auto_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    anchors: list[dict[str, float]] = []
    for row in rows:
        if any(row.get(key) in (None, "") for key in ("pdf_anchor_x_pt", "pdf_anchor_y_pt", "longitude", "latitude")):
            continue
        anchors.append(
            {
                "pdf_x": float(row["pdf_anchor_x_pt"]),
                "pdf_y": float(row["pdf_anchor_y_pt"]),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
            }
        )
    return anchors


def estimate_initial_corners(
    *,
    frame_row: dict[str, Any],
    manual_gcp: dict[str, Any] | None,
    auto_gcps: list[dict[str, Any]],
) -> tuple[dict[str, list[float]] | None, str]:
    candidate_rows = manual_anchor_rows(manual_gcp) if manual_gcp else []
    source_name = "manual_gcps"
    if len(candidate_rows) < 2:
        candidate_rows = auto_anchor_rows(auto_gcps)
        source_name = "auto_gcps"
    if len(candidate_rows) < 2:
        return None, "fallback_view"

    ref_lon = sum(row["longitude"] for row in candidate_rows) / len(candidate_rows)
    ref_lat = sum(row["latitude"] for row in candidate_rows) / len(candidate_rows)
    source = np.array([(row["pdf_x"], row["pdf_y"]) for row in candidate_rows], dtype=float)
    target = np.array(
        [lonlat_to_local_equirect(row["longitude"], row["latitude"], ref_lon, ref_lat) for row in candidate_rows],
        dtype=float,
    )
    try:
        if len(candidate_rows) >= 4:
            params = fit_projective(source, target)
            transformed_corners = apply_projective(
                np.array(
                    [
                        [float(frame_row["x0_pt"]), float(frame_row["y0_pt"])],
                        [float(frame_row["x1_pt"]), float(frame_row["y0_pt"])],
                        [float(frame_row["x1_pt"]), float(frame_row["y1_pt"])],
                        [float(frame_row["x0_pt"]), float(frame_row["y1_pt"])],
                    ],
                    dtype=float,
                ),
                params,
            )
            source_name = f"{source_name}_projective"
        else:
            params = fit_similarity(source, target)
            transformed_corners = apply_similarity(
                np.array(
                    [
                        [float(frame_row["x0_pt"]), float(frame_row["y0_pt"])],
                        [float(frame_row["x1_pt"]), float(frame_row["y0_pt"])],
                        [float(frame_row["x1_pt"]), float(frame_row["y1_pt"])],
                        [float(frame_row["x0_pt"]), float(frame_row["y1_pt"])],
                    ],
                    dtype=float,
                ),
                params,
            )
            source_name = f"{source_name}_similarity"
    except np.linalg.LinAlgError:
        return None, "fallback_view"

    lonlat_corners = [local_equirect_to_lonlat(x, y, ref_lon, ref_lat) for x, y in transformed_corners]
    return (
        {
            "top_left": [round(float(lonlat_corners[0][0]), 7), round(float(lonlat_corners[0][1]), 7)],
            "top_right": [round(float(lonlat_corners[1][0]), 7), round(float(lonlat_corners[1][1]), 7)],
            "bottom_right": [round(float(lonlat_corners[2][0]), 7), round(float(lonlat_corners[2][1]), 7)],
            "bottom_left": [round(float(lonlat_corners[3][0]), 7), round(float(lonlat_corners[3][1]), 7)],
        },
        source_name,
    )


def mean_center(points: list[tuple[float, float]]) -> list[float]:
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return [round(lon, 7), round(lat, 7)]


def zoom_from_extent(
    *,
    points: list[tuple[float, float]],
    image_width_px: int,
    image_height_px: int,
) -> float | None:
    if len(points) < 2:
        return None
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    center_lat = sum(lats) / len(lats)
    lon_span_m = max(lons) - min(lons)
    lon_span_m *= 111320.0 * max(0.1, math.cos(math.radians(center_lat)))
    lat_span_m = (max(lats) - min(lats)) * 110540.0
    span_m = max(lon_span_m, lat_span_m)
    if span_m <= 1e-6:
        return None
    target_px = max(320.0, max(image_width_px, image_height_px) * 1.35)
    meters_per_pixel = span_m / target_px
    world_mpp = 156543.03392 * max(0.1, math.cos(math.radians(center_lat)))
    zoom = math.log2(world_mpp / meters_per_pixel)
    return zoom


def zoom_from_frame_metrics(
    *,
    area_pt2: float,
    rmse_m: float | None,
) -> float:
    area_ratio = max(area_pt2, 1.0) / 100000.0
    zoom = 14.2 - (0.55 * math.log2(area_ratio))
    if rmse_m is not None and rmse_m > 0:
        zoom -= min(1.0, rmse_m / 80.0)
    return zoom


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_initial_view(
    *,
    frame_row: dict[str, Any],
    frame_model: dict[str, Any] | None,
    manual_gcp: dict[str, Any] | None,
    auto_gcps: list[dict[str, Any]],
    saved_georef: dict[str, Any] | None,
    image_width_px: int,
    image_height_px: int,
) -> tuple[list[float], float]:
    if saved_georef and saved_georef.get("corners_lonlat"):
        corners = saved_georef["corners_lonlat"]
        points = [
            tuple(corners[name])
            for name in ("top_left", "top_right", "bottom_right", "bottom_left")
            if name in corners
        ]
        if points:
            center = mean_center(points)
            zoom = zoom_from_extent(points=points, image_width_px=image_width_px, image_height_px=image_height_px)
            if zoom is not None:
                return center, round(clamp(zoom, 8.0, 17.0), 2)

    geo_points: list[tuple[float, float]] = []
    if frame_model and manual_gcp:
        geo_points = geo_points_from_manual_data(manual_gcp)
    if not geo_points and auto_gcps:
        geo_points = geo_points_from_auto_gcps(auto_gcps)

    if geo_points:
        center = mean_center(geo_points)
    else:
        center = DEFAULT_CENTER[:]

    zoom = zoom_from_extent(points=geo_points, image_width_px=image_width_px, image_height_px=image_height_px)
    if zoom is None:
        rmse_m = None
        if frame_model and frame_model.get("rmse_m") not in (None, ""):
            rmse_m = float(frame_model["rmse_m"])
        zoom = zoom_from_frame_metrics(area_pt2=float(frame_row["area_pt2"]), rmse_m=rmse_m)
    return center, round(clamp(zoom, 8.0, 17.0), 2)


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


def build_html(config: dict[str, Any]) -> str:
    template = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Overlay Georeferencing Editor</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
  <style>
    :root {
      --bg: #f5f4ef;
      --surface: #ffffff;
      --border: #d6d3d1;
      --text: #111827;
      --muted: #4b5563;
      --accent: #b91c1c;
      --accent-2: #1d4ed8;
      --ok: #166534;
      --warn: #92400e;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; color: var(--text); background: var(--bg); }
    #app { display: grid; grid-template-rows: auto auto 1fr; height: 100%; min-height: 0; }
    #toolbar {
      display: grid;
      grid-template-columns: 260px 170px 1fr auto auto auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.94);
    }
    #statusbar {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.9);
      align-items: center;
    }
    select, button {
      font: inherit;
      min-height: 38px;
      border-radius: 10px;
      border: 1px solid rgba(15,23,42,0.14);
      background: white;
      padding: 8px 10px;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: transparent; }
    #stepText {
      font-weight: 800;
      font-size: 18px;
      white-space: pre-wrap;
    }
    #metrics, #notice {
      font-size: 12px;
      white-space: pre-wrap;
      color: var(--muted);
    }
    #metrics strong { color: var(--text); }
    #workspace {
      display: grid;
      grid-template-columns: 1fr 1fr;
      min-height: 0;
    }
    .pane {
      position: relative;
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--border);
      background: #fff;
    }
    .pane:last-child { border-right: 0; }
    .paneTitle {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 5;
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(15,23,42,0.1);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    #pdfCanvas {
      position: absolute;
      inset: 0;
      cursor: crosshair;
    }
    #map {
      position: absolute;
      inset: 0;
    }
    #rightError {
      position: absolute;
      top: 50px;
      left: 12px;
      right: 12px;
      z-index: 6;
      background: rgba(127,29,29,0.95);
      color: white;
      padding: 10px 12px;
      border-radius: 12px;
      display: none;
      white-space: pre-wrap;
    }
    #jsonPreview {
      position: absolute;
      right: 12px;
      bottom: 12px;
      z-index: 6;
      width: min(420px, calc(100% - 24px));
      max-height: 40%;
      overflow: auto;
      background: rgba(15,23,42,0.9);
      color: #e2e8f0;
      border-radius: 12px;
      padding: 10px;
      font-size: 11px;
      line-height: 1.45;
      margin: 0;
    }
    @media (max-width: 1100px) {
      #toolbar {
        grid-template-columns: 1fr 1fr;
      }
      #workspace {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr 1fr;
      }
      .pane {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
      .pane:last-child { border-bottom: 0; }
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="toolbar">
      <select id="frameSelect"></select>
      <select id="modeSelect">
        <option value="similarity">2点合わせ</option>
        <option value="projective">4点合わせ</option>
      </select>
      <div></div>
      <button id="undoButton">Undo</button>
      <button id="clearButton">Clear</button>
      <button id="previewButton">Preview</button>
      <button id="saveButton" class="primary">Save</button>
      <button id="nextButton">Next Frame</button>
    </div>
    <div id="statusbar">
      <div id="stepText">次: 左の PDF をクリックしてください</div>
      <div id="metrics"></div>
      <div id="notice"></div>
    </div>
    <div id="workspace">
      <section class="pane">
        <div class="paneTitle">PDF</div>
        <canvas id="pdfCanvas"></canvas>
      </section>
      <section class="pane">
        <div class="paneTitle">Mapbox</div>
        <div id="map"></div>
        <div id="rightError"></div>
        <pre id="jsonPreview"></pre>
      </section>
    </div>
  </div>

  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <script>
    window.__OVERLAY_GEOREF_CONFIG__ = __CONFIG_JSON__;
  </script>
  <script>
    const config = window.__OVERLAY_GEOREF_CONFIG__;
    const MAPBOX_SCALE = 0.01;
    const EARTH_RADIUS_M = 6378137.0;
    const paneIds = { pdf: 'pdf', map: 'map' };
    const ui = {
      frameSelect: document.getElementById('frameSelect'),
      modeSelect: document.getElementById('modeSelect'),
      undoButton: document.getElementById('undoButton'),
      clearButton: document.getElementById('clearButton'),
      previewButton: document.getElementById('previewButton'),
      saveButton: document.getElementById('saveButton'),
      nextButton: document.getElementById('nextButton'),
      stepText: document.getElementById('stepText'),
      metrics: document.getElementById('metrics'),
      notice: document.getElementById('notice'),
      pdfCanvas: document.getElementById('pdfCanvas'),
      rightError: document.getElementById('rightError'),
      jsonPreview: document.getElementById('jsonPreview'),
    };
    const pdfCtx = ui.pdfCanvas.getContext('2d');
    const imageCache = new Map();
    const frameState = new Map();
    let currentFrameId = '';
    let map = null;
    let mapReady = false;

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function showRightError(message) {
      ui.rightError.style.display = 'block';
      ui.rightError.textContent = message;
    }

    function clearRightError() {
      ui.rightError.style.display = 'none';
      ui.rightError.textContent = '';
    }

    function validateConfig() {
      if (!config || !Array.isArray(config.frames)) {
        ui.stepText.textContent = 'manifest の読み込みに失敗しました';
        ui.notice.textContent = 'frames 配列がありません';
        return false;
      }
      if (config.frames.length === 0) {
        ui.stepText.textContent = 'frame がありません';
        ui.notice.textContent = 'frame_manifest.json の frames が空です';
        return false;
      }
      return true;
    }

    function sortedFrames() {
      return [...config.frames].sort((a, b) => {
        const diff = Number(b.priority_score || 0) - Number(a.priority_score || 0);
        return diff || (a.page_no - b.page_no) || a.frame_id.localeCompare(b.frame_id);
      });
    }

    function frameById(frameId) {
      return config.frames.find((frame) => frame.frame_id === frameId) || null;
    }

    function requiredPoints() {
      return ui.modeSelect.value === 'projective' ? 4 : 2;
    }

    function currentTransformType() {
      return ui.modeSelect.value;
    }

    function populateFrameSelect() {
      const frames = sortedFrames();
      ui.frameSelect.innerHTML = '';
      for (const frame of frames) {
        const option = document.createElement('option');
        option.value = frame.frame_id;
        option.textContent = `${frame.page_no} / ${frame.frame_id} / score=${Number(frame.priority_score || 0).toFixed(1)}`;
        ui.frameSelect.appendChild(option);
      }
      currentFrameId = config.default_frame_id || frames[0].frame_id;
      ui.frameSelect.value = currentFrameId;
    }

    function getState(frame) {
      if (!frameState.has(frame.frame_id)) {
        frameState.set(frame.frame_id, {
          controlPoints: [],
          preview: null,
          dirty: false,
          notice: frame.has_saved_georef ? '既存 saved georef あり' : '',
        });
      }
      return frameState.get(frame.frame_id);
    }

    function completedPairs(state) {
      return state.controlPoints.filter((pair) => pair.pdf_px && pair.lonlat).length;
    }

    function expectedSide(state) {
      const needed = requiredPoints();
      const completed = completedPairs(state);
      if (completed >= needed) {
        return 'done';
      }
      const last = state.controlPoints[state.controlPoints.length - 1];
      if (!last || (last.pdf_px && last.lonlat)) {
        return 'pdf';
      }
      return 'map';
    }

    function updateStepText(frame, state) {
      const side = expectedSide(state);
      const done = completedPairs(state);
      const needed = requiredPoints();
      if (side === 'done') {
        ui.stepText.textContent = `${needed}組の対応点が揃いました。Preview または Save できます`;
      } else if (side === 'pdf') {
        ui.stepText.textContent = `次: 左の PDF をクリックしてください (${done + 1}/${needed})`;
      } else {
        ui.stepText.textContent = `次: 右の Mapbox をクリックしてください (${done + 1}/${needed})`;
      }
      const rmse = state.preview && Number.isFinite(state.preview.rmse_m) ? `${state.preview.rmse_m.toFixed(2)} m` : '-';
      ui.metrics.innerHTML = `<strong>mode</strong>: ${currentTransformType()} / <strong>pairs</strong>: ${done}/${needed} / <strong>RMSE</strong>: ${rmse}`;
      ui.notice.textContent = state.notice || '';
    }

    async function loadImage(url) {
      if (imageCache.has(url)) {
        return imageCache.get(url);
      }
      const promise = new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error(`画像を読み込めません: ${url}`));
        image.src = url;
      });
      imageCache.set(url, promise);
      return promise;
    }

    function resizePdfCanvas() {
      const rect = ui.pdfCanvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      ui.pdfCanvas.width = Math.round(rect.width * dpr);
      ui.pdfCanvas.height = Math.round(rect.height * dpr);
      ui.pdfCanvas.style.width = `${rect.width}px`;
      ui.pdfCanvas.style.height = `${rect.height}px`;
      pdfCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return rect;
    }

    function fitImageRect(containerWidth, containerHeight, imageWidth, imageHeight) {
      const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight);
      const width = imageWidth * scale;
      const height = imageHeight * scale;
      return {
        x: (containerWidth - width) / 2,
        y: (containerHeight - height) / 2,
        width,
        height,
        scale,
      };
    }

    function drawMarker(ctx, x, y, label, fillColor, strokeColor, textColor) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.font = '700 12px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(label), x, y + 0.5);
      ctx.restore();
    }

    function pdfRectForFrame(frame, rect) {
      return fitImageRect(rect.width, rect.height, frame.image_width_px, frame.image_height_px);
    }

    function pdfPointToCanvas(pdfRect, point) {
      return {
        x: pdfRect.x + (point[0] * pdfRect.scale),
        y: pdfRect.y + (point[1] * pdfRect.scale),
      };
    }

    function canvasPointToPdf(pdfRect, x, y) {
      return [
        (x - pdfRect.x) / pdfRect.scale,
        (y - pdfRect.y) / pdfRect.scale,
      ];
    }

    async function drawPdfPane() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const rect = resizePdfCanvas();
      pdfCtx.clearRect(0, 0, rect.width, rect.height);
      pdfCtx.fillStyle = '#eef2f7';
      pdfCtx.fillRect(0, 0, rect.width, rect.height);
      const pdfRect = pdfRectForFrame(frame, rect);

      try {
        const baseImage = await loadImage(frame.image_path);
        pdfCtx.drawImage(baseImage, pdfRect.x, pdfRect.y, pdfRect.width, pdfRect.height);
        const redImage = await loadImage(frame.redlines_path);
        pdfCtx.globalAlpha = 0.95;
        pdfCtx.drawImage(redImage, pdfRect.x, pdfRect.y, pdfRect.width, pdfRect.height);
        pdfCtx.globalAlpha = 1;
      } catch (error) {
        pdfCtx.fillStyle = '#991b1b';
        pdfCtx.font = '700 14px ui-sans-serif, system-ui, sans-serif';
        pdfCtx.fillText(error.message, 20, 40);
      }

      pdfCtx.lineWidth = 3;
      pdfCtx.strokeStyle = 'rgba(29,78,216,0.95)';
      pdfCtx.strokeRect(pdfRect.x, pdfRect.y, pdfRect.width, pdfRect.height);

      state.controlPoints.forEach((pair, index) => {
        if (!pair.pdf_px) return;
        const point = pdfPointToCanvas(pdfRect, pair.pdf_px);
        drawMarker(pdfCtx, point.x, point.y, index + 1, 'rgba(29,78,216,0.95)', '#ffffff', '#ffffff');
      });

      return pdfRect;
    }

    function localMeters(lonlat, refLonLat) {
      const refLatRad = refLonLat[1] * Math.PI / 180;
      return [
        EARTH_RADIUS_M * ((lonlat[0] - refLonLat[0]) * Math.PI / 180) * Math.cos(refLatRad),
        EARTH_RADIUS_M * ((lonlat[1] - refLonLat[1]) * Math.PI / 180),
      ];
    }

    function metersToLonLat(point, refLonLat) {
      const refLatRad = refLonLat[1] * Math.PI / 180;
      return [
        refLonLat[0] + (point[0] / (EARTH_RADIUS_M * Math.cos(refLatRad))) * 180 / Math.PI,
        refLonLat[1] + (point[1] / EARTH_RADIUS_M) * 180 / Math.PI,
      ];
    }

    function solveLinearSystem(matrix, values) {
      const n = matrix.length;
      const a = matrix.map((row, idx) => [...row, values[idx]]);
      for (let col = 0; col < n; col += 1) {
        let pivot = col;
        for (let row = col + 1; row < n; row += 1) {
          if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
        }
        if (Math.abs(a[pivot][col]) < 1e-9) return null;
        if (pivot !== col) [a[pivot], a[col]] = [a[col], a[pivot]];
        const div = a[col][col];
        for (let k = col; k <= n; k += 1) a[col][k] /= div;
        for (let row = 0; row < n; row += 1) {
          if (row === col) continue;
          const factor = a[row][col];
          for (let k = col; k <= n; k += 1) a[row][k] -= factor * a[col][k];
        }
      }
      return a.map((row) => row[n]);
    }

    function fitProjective(src, dst) {
      const matrix = [];
      const values = [];
      for (let i = 0; i < src.length; i += 1) {
        const [x, y] = src[i];
        const [u, v] = dst[i];
        matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
        values.push(u);
        matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
        values.push(v);
      }
      return solveLinearSystem(matrix, values);
    }

    function applyProjectivePoint(params, point) {
      const [h11, h12, h13, h21, h22, h23, h31, h32] = params;
      const [x, y] = point;
      const denom = (h31 * x) + (h32 * y) + 1;
      if (Math.abs(denom) < 1e-9) return null;
      return [
        ((h11 * x) + (h12 * y) + h13) / denom,
        ((h21 * x) + (h22 * y) + h23) / denom,
      ];
    }

    function fitSimilarityFromTwoPairs(pairs) {
      const src1 = pairs[0].pdf_px;
      const src2 = pairs[1].pdf_px;
      const ref = [
        (pairs[0].lonlat[0] + pairs[1].lonlat[0]) / 2,
        (pairs[0].lonlat[1] + pairs[1].lonlat[1]) / 2,
      ];
      const dst1 = localMeters(pairs[0].lonlat, ref);
      const dst2 = localMeters(pairs[1].lonlat, ref);
      const sv = [src2[0] - src1[0], src2[1] - src1[1]];
      const tv = [dst2[0] - dst1[0], dst2[1] - dst1[1]];
      const sNorm = Math.hypot(sv[0], sv[1]);
      const tNorm = Math.hypot(tv[0], tv[1]);
      if (sNorm < 1e-6 || tNorm < 1e-6) {
        throw new Error('2点が近すぎるため similarity 変換を計算できません。');
      }
      const scale = tNorm / sNorm;
      const angle = Math.atan2(tv[1], tv[0]) - Math.atan2(sv[1], sv[0]);
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const srcCenter = [(src1[0] + src2[0]) / 2, (src1[1] + src2[1]) / 2];
      const dstCenter = [(dst1[0] + dst2[0]) / 2, (dst1[1] + dst2[1]) / 2];
      const rotatedCenter = [
        scale * ((cosA * srcCenter[0]) - (sinA * srcCenter[1])),
        scale * ((sinA * srcCenter[0]) + (cosA * srcCenter[1])),
      ];
      const translation = [dstCenter[0] - rotatedCenter[0], dstCenter[1] - rotatedCenter[1]];
      return {
        refLonLat: ref,
        apply(point) {
          const local = [
            scale * ((cosA * point[0]) - (sinA * point[1])) + translation[0],
            scale * ((sinA * point[0]) + (cosA * point[1])) + translation[1],
          ];
          return metersToLonLat(local, ref);
        },
      };
    }

    function fitProjectiveFromPairs(pairs) {
      const ref = pairs.reduce((acc, pair) => [acc[0] + pair.lonlat[0], acc[1] + pair.lonlat[1]], [0, 0]).map((value) => value / pairs.length);
      const src = pairs.map((pair) => pair.pdf_px);
      const dst = pairs.map((pair) => localMeters(pair.lonlat, ref));
      const params = fitProjective(src, dst);
      if (!params) {
        throw new Error('projective 変換を計算できません。');
      }
      return {
        refLonLat: ref,
        apply(point) {
          const local = applyProjectivePoint(params, point);
          if (!local) throw new Error('projective 変換結果が不正です。');
          return metersToLonLat(local, ref);
        },
      };
    }

    function imageCorners(frame) {
      return [
        [0, 0],
        [frame.image_width_px, 0],
        [frame.image_width_px, frame.image_height_px],
        [0, frame.image_height_px],
      ];
    }

    function routeCoordToPdfPx(frame, coord) {
      const bbox = frame.pdf_bbox;
      const pdfX = coord[0] / MAPBOX_SCALE;
      const pdfY = -coord[1] / MAPBOX_SCALE;
      const scaleX = frame.image_width_px / (bbox.x1 - bbox.x0);
      const scaleY = frame.image_height_px / (bbox.y1 - bbox.y0);
      return [
        (pdfX - bbox.x0) * scaleX,
        (pdfY - bbox.y0) * scaleY,
      ];
    }

    function transformRouteGeometry(frame, geometry, transformer) {
      const transformLine = (line) => line.map((coord) => transformer.apply(routeCoordToPdfPx(frame, coord)).map((value) => Number(value.toFixed(7))));
      if (geometry.type === 'LineString') {
        return { type: 'LineString', coordinates: transformLine(geometry.coordinates) };
      }
      return { type: 'MultiLineString', coordinates: geometry.coordinates.map(transformLine) };
    }

    function buildPreview(frame, state) {
      const pairs = state.controlPoints.filter((pair) => pair.pdf_px && pair.lonlat).slice(0, requiredPoints());
      if (pairs.length < requiredPoints()) {
        throw new Error(`${requiredPoints()}組の対応点が必要です。`);
      }
      const transformer = currentTransformType() === 'similarity'
        ? fitSimilarityFromTwoPairs(pairs)
        : fitProjectiveFromPairs(pairs);

      const transformedControl = [];
      const residuals = [];
      let sumSq = 0;
      for (let index = 0; index < pairs.length; index += 1) {
        const predicted = transformer.apply(pairs[index].pdf_px);
        transformedControl.push({ label: String(index + 1), lonlat: predicted });
        const targetMeters = localMeters(pairs[index].lonlat, transformer.refLonLat);
        const predMeters = localMeters(predicted, transformer.refLonLat);
        const error = Math.hypot(predMeters[0] - targetMeters[0], predMeters[1] - targetMeters[1]);
        sumSq += error * error;
        residuals.push({
          label: String(index + 1),
          coordinates: [predicted, pairs[index].lonlat],
          error_m: error,
        });
      }
      const rmse = Math.sqrt(sumSq / pairs.length);
      const corners = imageCorners(frame).map((point) => transformer.apply(point).map((value) => Number(value.toFixed(7))));
      const transformedRoutes = {
        type: 'FeatureCollection',
        features: frame.route_geojson.features.map((feature) => ({
          type: 'Feature',
          geometry: transformRouteGeometry(frame, feature.geometry, transformer),
          properties: { ...feature.properties },
        })),
      };
      return {
        transform_type: currentTransformType(),
        rmse_m: rmse,
        transformed_routes: transformedRoutes,
        transformed_control_points: transformedControl,
        residuals,
        corners_lonlat: {
          top_left: corners[0],
          top_right: corners[1],
          bottom_right: corners[2],
          bottom_left: corners[3],
        },
      };
    }

    function buildSavePayload(frame, state) {
      const preview = state.preview;
      if (!preview) {
        throw new Error('Preview を先に実行してください。');
      }
      return {
        page_no: frame.page_no,
        frame_id: frame.frame_id,
        transform_type: preview.transform_type,
        control_points: state.controlPoints
          .filter((pair) => pair.pdf_px && pair.lonlat)
          .slice(0, requiredPoints())
          .map((pair, index) => ({
            index: index + 1,
            pdf_image_px: {
              x: Number(pair.pdf_px[0].toFixed(3)),
              y: Number(pair.pdf_px[1].toFixed(3)),
            },
            map_lonlat: [
              Number(pair.lonlat[0].toFixed(7)),
              Number(pair.lonlat[1].toFixed(7)),
            ],
          })),
        corners_lonlat: preview.corners_lonlat,
        rmse_m: Number(preview.rmse_m.toFixed(3)),
        created_at: new Date().toISOString(),
      };
    }

    function downloadJson(filename, data) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    }

    function updateJsonPreview(frame, state) {
      try {
        ui.jsonPreview.textContent = JSON.stringify(buildSavePayload(frame, state), null, 2);
      } catch {
        ui.jsonPreview.textContent = JSON.stringify({
          page_no: frame.page_no,
          frame_id: frame.frame_id,
          transform_type: currentTransformType(),
          control_points: state.controlPoints,
        }, null, 2);
      }
    }

    function updateMapSources(frame, state) {
      if (!mapReady) return;
      const controlFeatures = [];
      const residualFeatures = [];
      state.controlPoints.forEach((pair, index) => {
        if (pair.lonlat) {
          controlFeatures.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: pair.lonlat },
            properties: { label: String(index + 1), kind: 'target' },
          });
        }
      });
      if (state.preview) {
        state.preview.transformed_control_points.forEach((point) => {
          controlFeatures.push({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: point.lonlat },
            properties: { label: point.label, kind: 'predicted' },
          });
        });
        state.preview.residuals.forEach((row) => {
          residualFeatures.push({
            type: 'Feature',
            geometry: { type: 'LineString', coordinates: row.coordinates },
            properties: { label: row.label, error_m: row.error_m },
          });
        });
      }
      map.getSource('routes')?.setData(state.preview ? state.preview.transformed_routes : { type: 'FeatureCollection', features: [] });
      map.getSource('control-points')?.setData({ type: 'FeatureCollection', features: controlFeatures });
      map.getSource('residuals')?.setData({ type: 'FeatureCollection', features: residualFeatures });
    }

    function fitPreviewBounds(frame, state) {
      if (!mapReady || !state.preview) return;
      const bounds = new mapboxgl.LngLatBounds();
      Object.values(state.preview.corners_lonlat).forEach((coord) => bounds.extend(coord));
      map.fitBounds(bounds, { padding: 40, duration: 0, maxZoom: 17 });
    }

    function updateUi() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      updateStepText(frame, state);
      updateJsonPreview(frame, state);
      updateMapSources(frame, state);
    }

    async function renderAll() {
      const frame = frameById(currentFrameId);
      if (!frame) return;
      await drawPdfPane();
      updateUi();
    }

    function previewCurrent() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      try {
        state.preview = buildPreview(frame, state);
        state.dirty = true;
        state.notice = `Preview 更新 / RMSE ${state.preview.rmse_m.toFixed(2)} m`;
        updateUi();
        fitPreviewBounds(frame, state);
      } catch (error) {
        state.notice = error.message;
        updateUi();
      }
    }

    function resetFrame(frame) {
      const state = getState(frame);
      state.controlPoints = [];
      state.preview = null;
      state.dirty = false;
      state.notice = '';
      renderAll();
    }

    function onPdfClick(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (expectedSide(state) !== 'pdf') return;
      const rect = ui.pdfCanvas.getBoundingClientRect();
      const pdfRect = pdfRectForFrame(frame, rect);
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (x < pdfRect.x || x > pdfRect.x + pdfRect.width || y < pdfRect.y || y > pdfRect.y + pdfRect.height) return;
      const point = canvasPointToPdf(pdfRect, x, y).map((value, index) => Number(clamp(value, 0, index === 0 ? frame.image_width_px : frame.image_height_px).toFixed(3)));
      state.controlPoints.push({ pdf_px: point, lonlat: null });
      state.preview = null;
      state.notice = `PDF point ${state.controlPoints.length} を記録しました`;
      renderAll();
    }

    function onMapClick(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (expectedSide(state) !== 'map') return;
      const pair = state.controlPoints[state.controlPoints.length - 1];
      if (!pair) return;
      pair.lonlat = [Number(event.lngLat.lng.toFixed(7)), Number(event.lngLat.lat.toFixed(7))];
      state.preview = null;
      state.notice = `Mapbox point ${completedPairs(state)} を記録しました`;
      if (completedPairs(state) >= requiredPoints()) {
        previewCurrent();
      } else {
        renderAll();
      }
    }

    function onUndo() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (!state.controlPoints.length) return;
      const last = state.controlPoints[state.controlPoints.length - 1];
      if (last.lonlat) {
        last.lonlat = null;
      } else {
        state.controlPoints.pop();
      }
      state.preview = null;
      state.notice = '最後の対応点を取り消しました';
      renderAll();
    }

    function onClear() {
      const frame = frameById(currentFrameId);
      resetFrame(frame);
    }

    function onSave() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      try {
        if (!state.preview) {
          state.preview = buildPreview(frame, state);
        }
        downloadJson(frame.suggested_download_filename, buildSavePayload(frame, state));
        state.dirty = false;
        state.notice = `保存用JSONをダウンロードしました\n配置先: data/manual_image_georef/\n${frame.suggested_download_filename}`;
        updateUi();
      } catch (error) {
        state.notice = error.message;
        updateUi();
      }
    }

    function onNextFrame() {
      const frames = sortedFrames();
      const index = frames.findIndex((frame) => frame.frame_id === currentFrameId);
      const next = frames[(index + 1) % frames.length];
      currentFrameId = next.frame_id;
      ui.frameSelect.value = currentFrameId;
      renderAll();
    }

    function onFrameChange() {
      currentFrameId = ui.frameSelect.value;
      renderAll();
    }

    function initMap() {
      if (!config.mapbox_access_token) {
        showRightError('MAPBOX_ACCESS_TOKEN がありません。.env を確認してください。');
        return;
      }
      if (typeof mapboxgl === 'undefined') {
        showRightError('Mapbox GL JS の読み込みに失敗しました。');
        return;
      }
      mapboxgl.accessToken = config.mapbox_access_token;
      map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v12',
        center: config.initial_map_center || [133.5, 33.8],
        zoom: config.initial_map_zoom || 8.5,
      });
      map.addControl(new mapboxgl.NavigationControl(), 'top-right');
      map.on('load', () => {
        mapReady = true;
        clearRightError();
        map.addSource('routes', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        map.addSource('control-points', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        map.addSource('residuals', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

        map.addLayer({
          id: 'routes-line',
          type: 'line',
          source: 'routes',
          paint: { 'line-color': '#dc2626', 'line-width': 3 },
        });
        map.addLayer({
          id: 'residuals-line',
          type: 'line',
          source: 'residuals',
          paint: { 'line-color': '#0f172a', 'line-width': 1.5, 'line-dasharray': [2, 2], 'line-opacity': 0.65 },
        });
        map.addLayer({
          id: 'control-target-circle',
          type: 'circle',
          source: 'control-points',
          filter: ['==', ['get', 'kind'], 'target'],
          paint: {
            'circle-radius': 8,
            'circle-color': '#f59e0b',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
          },
        });
        map.addLayer({
          id: 'control-predicted-circle',
          type: 'circle',
          source: 'control-points',
          filter: ['==', ['get', 'kind'], 'predicted'],
          paint: {
            'circle-radius': 8,
            'circle-color': '#2563eb',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
          },
        });
        map.addLayer({
          id: 'control-labels',
          type: 'symbol',
          source: 'control-points',
          layout: {
            'text-field': ['get', 'label'],
            'text-size': 12,
            'text-offset': [0, 0],
            'text-anchor': 'center',
          },
          paint: {
            'text-color': '#ffffff',
          },
        });
        map.on('click', onMapClick);
        map.on('error', (event) => {
          const message = event && event.error && event.error.message ? event.error.message : 'Mapbox error';
          showRightError(message);
        });
        renderAll();
      });
    }

    function wireUi() {
      ui.frameSelect.addEventListener('change', onFrameChange);
      ui.modeSelect.addEventListener('change', () => {
        const frame = frameById(currentFrameId);
        resetFrame(frame);
      });
      ui.undoButton.addEventListener('click', onUndo);
      ui.clearButton.addEventListener('click', onClear);
      ui.previewButton.addEventListener('click', previewCurrent);
      ui.saveButton.addEventListener('click', onSave);
      ui.nextButton.addEventListener('click', onNextFrame);
      ui.pdfCanvas.addEventListener('click', onPdfClick);
      window.addEventListener('resize', renderAll);
    }

    if (validateConfig()) {
      populateFrameSelect();
      wireUi();
      renderAll();
      initMap();
    }
  </script>
</body>
</html>
"""
    return template.replace("__CONFIG_JSON__", json.dumps(config, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build overlay georeferencing editor artifacts.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--frames", type=Path, default=Path("artifacts/step2/frames.csv"))
    parser.add_argument("--manual-review-queue", type=Path, default=Path("artifacts/step5/manual_review_queue.csv"))
    parser.add_argument("--frame-models", type=Path, default=Path("artifacts/step5/frame_models.csv"))
    parser.add_argument("--gcp-candidates", type=Path, default=Path("artifacts/step4/gcp_candidates.csv"))
    parser.add_argument("--routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--manual-gcp-dir", type=Path, default=Path("data/manual_gcps"))
    parser.add_argument("--saved-georef-dir", type=Path, default=Path("data/manual_image_georef"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/overlay_georef_editor"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--frame-id", action="append", default=[], help="Process only specific frame_id. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N frames after sorting.")
    parser.add_argument("--render-scale", type=float, default=RENDER_SCALE)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    image_dir = args.out_dir / "images"
    vector_dir = args.out_dir / "vectors"
    ensure_dir(image_dir)
    ensure_dir(vector_dir)
    ensure_dir(args.saved_georef_dir)

    frames = read_csv_rows(args.frames)
    priority_rows = priority_by_frame(args.manual_review_queue)
    frame_models = frame_models_by_frame(args.frame_models)
    auto_gcps = gcps_by_frame(args.gcp_candidates)
    manual_gcps = manual_gcps_by_frame(args.manual_gcp_dir)
    saved_georef = saved_georef_by_frame(args.saved_georef_dir)
    routes_by_frame = route_features_by_frame(args.routes)
    token = read_mapbox_token(args.env_file)

    selected_frame_ids = set(args.frame_id)
    frame_rows: list[dict[str, Any]] = []
    for frame_row in frames:
        if selected_frame_ids and frame_row["frame_id"] not in selected_frame_ids:
            continue
        frame_rows.append(frame_row)

    frame_rows.sort(
        key=lambda row: (
            -float((priority_rows.get((int(row["page_no"]), row["frame_id"]), {}) or {}).get("priority_score", 0) or 0),
            int(row["page_no"]),
            row["frame_id"],
        )
    )
    if args.limit > 0:
        frame_rows = frame_rows[:args.limit]

    manifest_frames: list[dict[str, Any]] = []
    doc = fitz.open(args.pdf)
    try:
        for frame_row in frame_rows:
            page_no = int(frame_row["page_no"])
            frame_id = frame_row["frame_id"]
            key = (page_no, frame_id)
            map_image, redlines_image = render_frame_images(
                doc=doc,
                page_no=page_no,
                frame_row=frame_row,
                render_scale=args.render_scale,
            )
            map_path = image_dir / f"{frame_id}_map.png"
            redlines_path = image_dir / f"{frame_id}_redlines.png"
            map_image.save(map_path)
            redlines_image.save(redlines_path)

            features = routes_by_frame.get(key, [])
            vector_path = vector_dir / f"{frame_id}_routes.geojson"
            save_feature_collection(vector_path, features)

            frame_model = frame_models.get(key)
            manual_gcp = manual_gcps.get(key)
            auto_rows = auto_gcps.get(key, [])
            saved_row = saved_georef.get(key)
            saved_corners = saved_row.get("corners_lonlat") if saved_row else None
            initial_corners, initial_transform_source = estimate_initial_corners(
                frame_row=frame_row,
                manual_gcp=manual_gcp,
                auto_gcps=auto_rows,
            )
            initial_center, initial_zoom = choose_initial_view(
                frame_row=frame_row,
                frame_model=frame_model,
                manual_gcp=manual_gcp,
                auto_gcps=auto_rows,
                saved_georef=saved_row,
                image_width_px=map_image.width,
                image_height_px=map_image.height,
            )
            priority_row = priority_rows.get(key, {})
            corners_for_view = saved_corners or initial_corners
            if corners_for_view:
                points = corners_dict_to_points(corners_for_view)
                if points:
                    initial_center = mean_center(points)
                    zoom = zoom_from_extent(points=points, image_width_px=map_image.width, image_height_px=map_image.height)
                    if zoom is not None:
                        initial_zoom = round(clamp(zoom, 8.0, 17.0), 2)
            manifest_frames.append(
                {
                    "frame_id": frame_id,
                    "page_no": page_no,
                    "priority_score": round(float(priority_row.get("priority_score", 0) or 0), 3),
                    "priority_reason": priority_row.get("priority_reason", ""),
                    "route_count": len(features),
                    "image_path": f"images/{frame_id}_map.png",
                    "redlines_path": f"images/{frame_id}_redlines.png",
                    "vector_geojson_path": f"vectors/{frame_id}_routes.geojson",
                    "route_geojson": {"type": "FeatureCollection", "features": features},
                    "pdf_bbox": rounded_bbox(frame_row),
                    "image_width_px": map_image.width,
                    "image_height_px": map_image.height,
                    "initial_center": initial_center,
                    "initial_zoom": initial_zoom,
                    "initial_corners": initial_corners,
                    "initial_transform_source": "saved_georef" if saved_corners else initial_transform_source,
                    "has_saved_georef": bool(saved_corners),
                    "saved_corners": saved_corners,
                    "saved_georef_path": str(saved_row.get("__path__", "")) if saved_row else "",
                    "suggested_download_filename": f"page_{page_no:03d}_{frame_id}.json",
                    "frame_area_pt2": round(float(frame_row["area_pt2"]), 3),
                    "rmse_m": None if not frame_model or frame_model.get("rmse_m") in (None, "") else round(float(frame_model["rmse_m"]), 3),
                    "quality_status": (frame_model or {}).get("quality_status", ""),
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
        "default_frame_id": manifest_frames[0]["frame_id"] if manifest_frames else "",
        "frames": manifest_frames,
    }

    (args.out_dir / "frame_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(build_html(manifest), encoding="utf-8")

    image_count = len(list(image_dir.glob("*_map.png"))) + len(list(image_dir.glob("*_redlines.png")))
    vector_count = len(list(vector_dir.glob("*_routes.geojson")))
    print(f"Wrote {len(manifest_frames)} frames to {args.out_dir}")
    print("Sanity check:")
    print(f"  frame_count={len(manifest_frames)}")
    print(f"  image_count={image_count}")
    print(f"  route_geojson_count={vector_count}")
    print(f"  default_frame={manifest['default_frame_id'] or '(empty)'}")


if __name__ == "__main__":
    main()
