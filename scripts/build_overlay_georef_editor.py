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


MAPBOX_SCALE = 0.01
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
      --panel-bg: rgba(255, 255, 255, 0.96);
      --panel-border: rgba(15, 23, 42, 0.12);
      --accent: #be123c;
      --text: #0f172a;
      --muted: #475569;
      --ok: #15803d;
      --warn: #92400e;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; color: var(--text); }
    body { background: #e2e8f0; }
    #app { display: grid; grid-template-rows: auto 1fr; height: 100%; }
    #controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      padding: 10px 14px;
      background: var(--panel-bg);
      border-bottom: 1px solid var(--panel-border);
      backdrop-filter: blur(10px);
      position: relative;
      z-index: 5;
    }
    #controls label, #controls select, #controls button, #controls input {
      font: inherit;
    }
    #controls .group {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
    }
    #controls select, #controls button, #controls input[type="range"] {
      min-height: 34px;
    }
    #controls select, #controls button {
      border: 1px solid var(--panel-border);
      background: white;
      border-radius: 8px;
      padding: 6px 10px;
    }
    #controls button.primary {
      background: var(--accent);
      color: white;
      border-color: transparent;
    }
    #controls .status {
      margin-left: auto;
      font-size: 13px;
      color: var(--muted);
    }
    #controls .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(21, 128, 61, 0.12);
      color: var(--ok);
    }
    #controls .badge.warn {
      background: rgba(146, 64, 14, 0.12);
      color: var(--warn);
    }
    #mapWrap {
      position: relative;
      min-height: 0;
    }
    #map {
      position: absolute;
      inset: 0;
    }
    #overlayCanvas {
      position: absolute;
      inset: 0;
      z-index: 2;
      pointer-events: none;
    }
    .corner-handle {
      position: absolute;
      width: 16px;
      height: 16px;
      margin-left: -8px;
      margin-top: -8px;
      border-radius: 50%;
      border: 2px solid white;
      background: var(--accent);
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
      z-index: 4;
      cursor: grab;
      display: none;
    }
    .corner-handle.visible {
      display: block;
    }
    #message {
      position: absolute;
      left: 14px;
      bottom: 14px;
      z-index: 6;
      background: rgba(15, 23, 42, 0.82);
      color: white;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      max-width: min(420px, calc(100vw - 28px));
      white-space: pre-wrap;
      pointer-events: none;
    }
    #error {
      position: absolute;
      inset: 12px 12px auto 12px;
      z-index: 7;
      background: rgba(127, 29, 29, 0.94);
      color: white;
      padding: 10px 12px;
      border-radius: 10px;
      display: none;
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="controls">
      <div class="group">
        <label for="frameOrder">順序</label>
        <select id="frameOrder">
          <option value="priority">優先度順</option>
          <option value="page">ページ順</option>
        </select>
      </div>
      <div class="group">
        <label for="frameSelect">Frame</label>
        <select id="frameSelect"></select>
      </div>
      <div class="group">
        <label for="opacityRange">Opacity</label>
        <input id="opacityRange" type="range" min="0.05" max="1" step="0.05" value="0.65" />
      </div>
      <div class="group">
        <label for="displayMode">表示</label>
        <select id="displayMode">
          <option value="both">Both</option>
          <option value="map">Map</option>
          <option value="redlines">RedLine</option>
        </select>
      </div>
      <div class="group">
        <label for="editMode">操作</label>
        <select id="editMode">
          <option value="move">Move</option>
          <option value="fine">Fine-tune</option>
        </select>
      </div>
      <div class="group">
        <button id="saveButton" class="primary">Save</button>
        <button id="resetButton">Reset</button>
        <button id="nextButton">Next Frame</button>
      </div>
      <div class="status" id="statusText">未保存</div>
    </div>
    <div id="mapWrap">
      <div id="map"></div>
      <canvas id="overlayCanvas"></canvas>
      <div id="handle0" class="corner-handle" data-index="0"></div>
      <div id="handle1" class="corner-handle" data-index="1"></div>
      <div id="handle2" class="corner-handle" data-index="2"></div>
      <div id="handle3" class="corner-handle" data-index="3"></div>
      <div id="error"></div>
      <div id="message"></div>
    </div>
  </div>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <script>
    window.__OVERLAY_GEOREF_CONFIG__ = __CONFIG_JSON__;
  </script>
  <script>
    const config = window.__OVERLAY_GEOREF_CONFIG__;
    const cornersOrder = ['top_left', 'top_right', 'bottom_right', 'bottom_left'];
    const frameState = new Map();
    const imageCache = new Map();
    const ui = {
      frameOrder: document.getElementById('frameOrder'),
      frameSelect: document.getElementById('frameSelect'),
      opacityRange: document.getElementById('opacityRange'),
      displayMode: document.getElementById('displayMode'),
      editMode: document.getElementById('editMode'),
      saveButton: document.getElementById('saveButton'),
      resetButton: document.getElementById('resetButton'),
      nextButton: document.getElementById('nextButton'),
      statusText: document.getElementById('statusText'),
      message: document.getElementById('message'),
      error: document.getElementById('error'),
      mapWrap: document.getElementById('mapWrap'),
      canvas: document.getElementById('overlayCanvas'),
      handles: [0, 1, 2, 3].map((index) => document.getElementById(`handle${index}`)),
    };

    const ctx = ui.canvas.getContext('2d');
    let currentFrameId = config.default_frame_id || (config.frames[0] && config.frames[0].frame_id) || '';
    let map = null;
    let pointerState = null;

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function showError(message) {
      ui.error.style.display = 'block';
      ui.error.textContent = message;
    }

    function clearError() {
      ui.error.style.display = 'none';
      ui.error.textContent = '';
    }

    function frameById(frameId) {
      return config.frames.find((frame) => frame.frame_id === frameId) || null;
    }

    function sortedFrames() {
      const frames = [...config.frames];
      if (ui.frameOrder.value === 'page') {
        frames.sort((a, b) => (a.page_no - b.page_no) || a.frame_id.localeCompare(b.frame_id));
      } else {
        frames.sort((a, b) => {
          const score = (Number(b.priority_score || 0) - Number(a.priority_score || 0));
          return score || ((a.page_no - b.page_no) || a.frame_id.localeCompare(b.frame_id));
        });
      }
      return frames;
    }

    function repopulateFrameSelect() {
      const frames = sortedFrames();
      const selected = currentFrameId || (frames[0] && frames[0].frame_id) || '';
      ui.frameSelect.innerHTML = '';
      for (const frame of frames) {
        const option = document.createElement('option');
        option.value = frame.frame_id;
        const savedLabel = frame.has_saved_georef ? ' [saved]' : '';
        option.textContent = `${frame.page_no} / ${frame.frame_id} / score=${Number(frame.priority_score || 0).toFixed(1)}${savedLabel}`;
        ui.frameSelect.appendChild(option);
      }
      ui.frameSelect.value = selected;
      currentFrameId = ui.frameSelect.value || selected;
    }

    function getFrameCenter(corners) {
      const lon = corners.reduce((sum, point) => sum + point[0], 0) / corners.length;
      const lat = corners.reduce((sum, point) => sum + point[1], 0) / corners.length;
      return [lon, lat];
    }

    function metersPerPixel(lat, zoom) {
      return 156543.03392 * Math.cos((lat * Math.PI) / 180) / Math.pow(2, zoom);
    }

    function offsetLonLat(center, dxMeters, dyMeters) {
      const lat = center[1] + (dyMeters / 6378137.0) * (180 / Math.PI);
      const lon = center[0] + (dxMeters / (6378137.0 * Math.cos((center[1] * Math.PI) / 180))) * (180 / Math.PI);
      return [lon, lat];
    }

    function cornersFromInitialView(frame) {
      const center = frame.initial_center || [133.5, 33.8];
      const zoom = Number(frame.initial_zoom || 12);
      const mpp = metersPerPixel(center[1], zoom);
      const widthMeters = frame.image_width_px * mpp;
      const heightMeters = frame.image_height_px * mpp;
      const halfW = widthMeters / 2;
      const halfH = heightMeters / 2;
      return [
        offsetLonLat(center, -halfW, halfH),
        offsetLonLat(center, halfW, halfH),
        offsetLonLat(center, halfW, -halfH),
        offsetLonLat(center, -halfW, -halfH),
      ];
    }

    function cornersFromSaved(savedCorners) {
      return cornersOrder.map((name) => savedCorners[name].slice());
    }

    function estimateZoomFromCorners(corners, frame) {
      const center = getFrameCenter(corners);
      const lons = corners.map((point) => point[0]);
      const lats = corners.map((point) => point[1]);
      const lonSpanMeters = (Math.max(...lons) - Math.min(...lons)) * 111320.0 * Math.max(0.1, Math.cos((center[1] * Math.PI) / 180));
      const latSpanMeters = (Math.max(...lats) - Math.min(...lats)) * 110540.0;
      const spanMeters = Math.max(lonSpanMeters, latSpanMeters, 1);
      const targetPx = Math.max(frame.image_width_px, frame.image_height_px) * 1.4;
      const mpp = spanMeters / targetPx;
      const zoom = Math.log2((156543.03392 * Math.max(0.1, Math.cos((center[1] * Math.PI) / 180))) / mpp);
      return clamp(zoom, 8, 17);
    }

    function getState(frame) {
      if (!frameState.has(frame.frame_id)) {
        const corners = frame.saved_corners ? cornersFromSaved(frame.saved_corners) : cornersFromInitialView(frame);
        frameState.set(frame.frame_id, {
          corners,
          dirty: false,
          saveState: frame.has_saved_georef ? 'saved' : 'unsaved',
        });
      }
      return frameState.get(frame.frame_id);
    }

    function setStatus(frame, state) {
      const labels = [];
      labels.push(`Frame ${frame.frame_id}`);
      labels.push(state.dirty ? '未保存の変更あり' : (state.saveState === 'saved' ? '保存済み' : '未保存'));
      if (frame.has_saved_georef) {
        labels.push('<span class="badge">保存済み JSON あり</span>');
      } else {
        labels.push('<span class="badge warn">保存データなし</span>');
      }
      ui.statusText.innerHTML = labels.join(' / ');
    }

    function setMessage(text) {
      ui.message.textContent = text;
    }

    function resizeCanvas() {
      const rect = ui.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      ui.canvas.width = Math.round(rect.width * dpr);
      ui.canvas.height = Math.round(rect.height * dpr);
      ui.canvas.style.width = `${rect.width}px`;
      ui.canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function projectedCorners(corners) {
      return corners.map((point) => map.project(point));
    }

    function drawPolygonOutline(screenCorners) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(screenCorners[0].x, screenCorners[0].y);
      for (let i = 1; i < screenCorners.length; i += 1) {
        ctx.lineTo(screenCorners[i].x, screenCorners[i].y);
      }
      ctx.closePath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(190,18,60,0.95)';
      ctx.setLineDash([8, 6]);
      ctx.stroke();
      ctx.restore();
    }

    function solveAffine(src, dst) {
      const [[sx1, sy1], [sx2, sy2], [sx3, sy3]] = src;
      const [[dx1, dy1], [dx2, dy2], [dx3, dy3]] = dst;
      const denom = (sx1 * (sy2 - sy3)) + (sx2 * (sy3 - sy1)) + (sx3 * (sy1 - sy2));
      if (Math.abs(denom) < 1e-9) {
        return null;
      }
      const a = ((dx1 * (sy2 - sy3)) + (dx2 * (sy3 - sy1)) + (dx3 * (sy1 - sy2))) / denom;
      const b = ((dy1 * (sy2 - sy3)) + (dy2 * (sy3 - sy1)) + (dy3 * (sy1 - sy2))) / denom;
      const c = ((dx1 * (sx3 - sx2)) + (dx2 * (sx1 - sx3)) + (dx3 * (sx2 - sx1))) / denom;
      const d = ((dy1 * (sx3 - sx2)) + (dy2 * (sx1 - sx3)) + (dy3 * (sx2 - sx1))) / denom;
      const e = ((dx1 * ((sx2 * sy3) - (sx3 * sy2))) + (dx2 * ((sx3 * sy1) - (sx1 * sy3))) + (dx3 * ((sx1 * sy2) - (sx2 * sy1)))) / denom;
      const f = ((dy1 * ((sx2 * sy3) - (sx3 * sy2))) + (dy2 * ((sx3 * sy1) - (sx1 * sy3))) + (dy3 * ((sx1 * sy2) - (sx2 * sy1)))) / denom;
      return [a, b, c, d, e, f];
    }

    function drawTriangleImage(image, srcTriangle, dstTriangle, opacity) {
      const transform = solveAffine(srcTriangle, dstTriangle);
      if (!transform) {
        return;
      }
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(dstTriangle[0][0], dstTriangle[0][1]);
      ctx.lineTo(dstTriangle[1][0], dstTriangle[1][1]);
      ctx.lineTo(dstTriangle[2][0], dstTriangle[2][1]);
      ctx.closePath();
      ctx.clip();
      ctx.globalAlpha = opacity;
      ctx.setTransform(...transform);
      ctx.drawImage(image, 0, 0);
      ctx.restore();
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function drawWarpedImage(image, screenCorners, opacity) {
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      const dst = screenCorners.map((point) => [point.x, point.y]);
      drawTriangleImage(image, [[0, 0], [width, 0], [width, height]], [dst[0], dst[1], dst[2]], opacity);
      drawTriangleImage(image, [[0, 0], [width, height], [0, height]], [dst[0], dst[2], dst[3]], opacity);
    }

    function pointInPolygon(point, polygon) {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
        const xi = polygon[i].x;
        const yi = polygon[i].y;
        const xj = polygon[j].x;
        const yj = polygon[j].y;
        const intersect = ((yi > point.y) !== (yj > point.y))
          && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-9) + xi);
        if (intersect) {
          inside = !inside;
        }
      }
      return inside;
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

    async function renderOverlay() {
      if (!map) {
        return;
      }
      const frame = frameById(currentFrameId);
      if (!frame) {
        return;
      }
      const state = getState(frame);
      resizeCanvas();
      ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
      const screenCorners = projectedCorners(state.corners);
      const opacity = Number(ui.opacityRange.value || 0.65);
      const displayMode = ui.displayMode.value;
      try {
        if (displayMode === 'map' || displayMode === 'both') {
          const image = await loadImage(frame.image_path);
          drawWarpedImage(image, screenCorners, opacity);
        }
        if (displayMode === 'redlines' || displayMode === 'both') {
          const image = await loadImage(frame.redlines_path);
          drawWarpedImage(image, screenCorners, displayMode === 'both' ? Math.min(1, opacity + 0.15) : opacity);
        }
      } catch (error) {
        showError(error.message);
      }
      drawPolygonOutline(screenCorners);
      updateHandles(screenCorners);
      setStatus(frame, state);
      setMessage([
        `page=${frame.page_no} frame=${frame.frame_id}`,
        `priority=${Number(frame.priority_score || 0).toFixed(1)} routes=${frame.route_count}`,
        'Move: drag / Shift+drag rotate / wheel scale',
        'Fine-tune: 四隅ハンドルをドラッグ',
      ].join('\\n'));
    }

    function updateHandles(screenCorners) {
      const visible = ui.editMode.value === 'fine';
      ui.handles.forEach((handle, index) => {
        handle.classList.toggle('visible', visible);
        if (visible) {
          handle.style.left = `${screenCorners[index].x}px`;
          handle.style.top = `${screenCorners[index].y}px`;
        }
      });
    }

    function updateFrameView(frame) {
      const state = getState(frame);
      const center = getFrameCenter(state.corners);
      const zoom = estimateZoomFromCorners(state.corners, frame);
      map.jumpTo({ center, zoom });
      renderOverlay();
    }

    function markDirty() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      state.dirty = true;
      state.saveState = 'unsaved';
      setStatus(frame, state);
    }

    function pointerPosition(event) {
      const rect = ui.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function screenToLonLat(point) {
      const lngLat = map.unproject([point.x, point.y]);
      return [lngLat.lng, lngLat.lat];
    }

    function translateCorners(baseScreenCorners, dx, dy) {
      return baseScreenCorners.map((point) => screenToLonLat({ x: point.x + dx, y: point.y + dy }));
    }

    function rotateCorners(baseScreenCorners, center, angle) {
      return baseScreenCorners.map((point) => {
        const x = point.x - center.x;
        const y = point.y - center.y;
        const rx = (x * Math.cos(angle)) - (y * Math.sin(angle));
        const ry = (x * Math.sin(angle)) + (y * Math.cos(angle));
        return screenToLonLat({ x: center.x + rx, y: center.y + ry });
      });
    }

    function scaleCorners(baseScreenCorners, center, factor) {
      return baseScreenCorners.map((point) => {
        const x = center.x + ((point.x - center.x) * factor);
        const y = center.y + ((point.y - center.y) * factor);
        return screenToLonLat({ x, y });
      });
    }

    function onCanvasPointerDown(event) {
      if (ui.editMode.value !== 'move') {
        return;
      }
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const screenCorners = projectedCorners(state.corners);
      const pos = pointerPosition(event);
      if (!pointInPolygon(pos, screenCorners)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const center = screenCorners.reduce((acc, point) => ({ x: acc.x + point.x / 4, y: acc.y + point.y / 4 }), { x: 0, y: 0 });
      const startAngle = Math.atan2(pos.y - center.y, pos.x - center.x);
      pointerState = {
        kind: event.shiftKey ? 'rotate' : 'move',
        start: pos,
        center,
        baseScreenCorners: screenCorners,
        startAngle,
      };
      map.dragPan.disable();
    }

    function onPointerMove(event) {
      if (!pointerState) {
        return;
      }
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const pos = pointerPosition(event);
      if (pointerState.kind === 'move') {
        state.corners = translateCorners(pointerState.baseScreenCorners, pos.x - pointerState.start.x, pos.y - pointerState.start.y);
      } else if (pointerState.kind === 'rotate') {
        const angle = Math.atan2(pos.y - pointerState.center.y, pos.x - pointerState.center.x) - pointerState.startAngle;
        state.corners = rotateCorners(pointerState.baseScreenCorners, pointerState.center, angle);
      } else if (pointerState.kind === 'fine') {
        state.corners[pointerState.cornerIndex] = screenToLonLat(pos);
      }
      markDirty();
      renderOverlay();
    }

    function onPointerUp() {
      if (!pointerState) {
        return;
      }
      pointerState = null;
      map.dragPan.enable();
    }

    function onWheel(event) {
      if (ui.editMode.value !== 'move') {
        return;
      }
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const screenCorners = projectedCorners(state.corners);
      const pos = pointerPosition(event);
      if (!pointInPolygon(pos, screenCorners)) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const center = screenCorners.reduce((acc, point) => ({ x: acc.x + point.x / 4, y: acc.y + point.y / 4 }), { x: 0, y: 0 });
      const factor = Math.exp(-event.deltaY * 0.0015);
      state.corners = scaleCorners(screenCorners, center, factor);
      markDirty();
      renderOverlay();
    }

    function activateFineHandle(handle, index) {
      handle.addEventListener('pointerdown', (event) => {
        if (ui.editMode.value !== 'fine') {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        pointerState = { kind: 'fine', cornerIndex: index };
        map.dragPan.disable();
      });
    }

    function buildSavePayload(frame, corners) {
      return {
        page_no: frame.page_no,
        frame_id: frame.frame_id,
        image_width_px: frame.image_width_px,
        image_height_px: frame.image_height_px,
        pdf_frame_bbox: frame.pdf_bbox,
        corners_lonlat: {
          top_left: corners[0].map((value) => Number(value.toFixed(7))),
          top_right: corners[1].map((value) => Number(value.toFixed(7))),
          bottom_right: corners[2].map((value) => Number(value.toFixed(7))),
          bottom_left: corners[3].map((value) => Number(value.toFixed(7))),
        },
        transform_type: 'projective',
        created_at: new Date().toISOString(),
      };
    }

    function downloadJson(filename, data) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function onSave() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const payload = buildSavePayload(frame, state.corners);
      downloadJson(frame.suggested_download_filename, payload);
      state.dirty = false;
      state.saveState = 'saved';
      setStatus(frame, state);
      setMessage(`保存用 JSON をダウンロードしました\\n${frame.suggested_download_filename}\\n配置先: data/manual_image_georef/`);
    }

    function onReset() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      state.corners = frame.saved_corners ? cornersFromSaved(frame.saved_corners) : cornersFromInitialView(frame);
      state.dirty = false;
      state.saveState = frame.has_saved_georef ? 'saved' : 'unsaved';
      updateFrameView(frame);
    }

    function onNextFrame() {
      const frames = sortedFrames();
      const index = frames.findIndex((frame) => frame.frame_id === currentFrameId);
      const next = frames[(index + 1) % frames.length];
      if (next) {
        currentFrameId = next.frame_id;
        ui.frameSelect.value = next.frame_id;
        updateFrameView(next);
      }
    }

    function onFrameChange() {
      currentFrameId = ui.frameSelect.value;
      const frame = frameById(currentFrameId);
      if (frame) {
        updateFrameView(frame);
      }
    }

    function wireUi() {
      ui.frameOrder.addEventListener('change', () => {
        const current = currentFrameId;
        repopulateFrameSelect();
        currentFrameId = current;
        ui.frameSelect.value = currentFrameId;
      });
      ui.frameSelect.addEventListener('change', onFrameChange);
      ui.opacityRange.addEventListener('input', renderOverlay);
      ui.displayMode.addEventListener('change', renderOverlay);
      ui.editMode.addEventListener('change', renderOverlay);
      ui.saveButton.addEventListener('click', onSave);
      ui.resetButton.addEventListener('click', onReset);
      ui.nextButton.addEventListener('click', onNextFrame);
      ui.mapWrap.addEventListener('pointerdown', onCanvasPointerDown, true);
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
      ui.mapWrap.addEventListener('wheel', onWheel, { passive: false, capture: true });
      ui.handles.forEach((handle, index) => activateFineHandle(handle, index));
    }

    function initMap() {
      if (!config.mapbox_access_token) {
        showError('MAPBOX_ACCESS_TOKEN が見つかりません。.env に設定してから build を再実行してください。');
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
        clearError();
        repopulateFrameSelect();
        const initial = frameById(currentFrameId) || config.frames[0];
        if (initial) {
          updateFrameView(initial);
        }
      });
      map.on('move', renderOverlay);
      map.on('resize', renderOverlay);
    }

    wireUi();
    initMap();
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
            saved_corners = saved_row.get("corners_lonlat") if saved_row else None
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
                    "pdf_bbox": rounded_bbox(frame_row),
                    "image_width_px": map_image.width,
                    "image_height_px": map_image.height,
                    "initial_center": initial_center,
                    "initial_zoom": initial_zoom,
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

    print(f"Wrote {len(manifest_frames)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
