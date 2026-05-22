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
      --panel-bg: #fcfcfa;
      --panel-border: #d6d3d1;
      --surface: rgba(255,255,255,0.92);
      --accent: #b91c1c;
      --accent-2: #0f766e;
      --text: #111827;
      --muted: #4b5563;
      --ok: #166534;
      --warn: #92400e;
      --shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; color: var(--text); }
    body { background: linear-gradient(180deg, #ece7e1 0%, #dbe4ea 100%); }
    #app {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100%;
      min-height: 0;
    }
    #sidebar {
      border-right: 1px solid var(--panel-border);
      background: var(--panel-bg);
      overflow: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .panel {
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 14px;
      padding: 12px;
      box-shadow: var(--shadow);
    }
    .panel h2, .panel h3, .panel p { margin: 0; }
    .panel h2 { font-size: 15px; margin-bottom: 10px; }
    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 10px;
    }
    .field:last-child { margin-bottom: 0; }
    label { font-size: 12px; color: var(--muted); font-weight: 700; letter-spacing: 0.02em; }
    select, button, input {
      font: inherit;
    }
    select, input[type="range"], button {
      width: 100%;
    }
    select, button {
      min-height: 36px;
      border-radius: 10px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      background: white;
      padding: 8px 10px;
    }
    button {
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: white;
      border-color: transparent;
    }
    button.secondary {
      background: #f5f5f4;
    }
    .button-row {
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr 1fr;
    }
    .button-row.triple {
      grid-template-columns: 1fr 1fr 1fr;
    }
    .mode-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr 1fr;
    }
    .mode-button.active {
      background: #111827;
      color: white;
      border-color: #111827;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(15, 118, 110, 0.12);
      color: var(--accent-2);
    }
    .badge.warn {
      background: rgba(146, 64, 14, 0.12);
      color: var(--warn);
    }
    #modeLabel {
      font-size: 22px;
      font-weight: 800;
      line-height: 1.1;
      margin-bottom: 8px;
    }
    #stepText {
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.5;
    }
    #statusText, #saveHint, #frameInfo {
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.5;
      color: var(--muted);
    }
    #jsonPreview {
      margin: 0;
      font-size: 11px;
      line-height: 1.4;
      overflow: auto;
      max-height: 240px;
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px;
      border-radius: 12px;
    }
    .hint {
      font-size: 11px;
      line-height: 1.5;
      color: var(--muted);
      white-space: pre-wrap;
    }
    #mapWrap {
      position: relative;
      min-width: 0;
      min-height: 0;
    }
    #map, #overlayCanvas {
      position: absolute;
      inset: 0;
    }
    #overlayCanvas {
      z-index: 2;
      pointer-events: none;
    }
    .overlay-handle {
      position: absolute;
      z-index: 4;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      pointer-events: auto;
      user-select: none;
      touch-action: none;
    }
    .corner-handle {
      width: 16px;
      height: 16px;
      background: rgba(185, 28, 28, 0.35);
      border: 2px solid rgba(255,255,255,0.95);
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.2);
      cursor: grab;
      opacity: 0.7;
    }
    .corner-handle.active {
      opacity: 1;
      background: rgba(185, 28, 28, 0.95);
    }
    #centerHandle {
      width: 34px;
      height: 34px;
      background: rgba(15, 118, 110, 0.9);
      border: 3px solid rgba(255,255,255,0.96);
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.28);
      cursor: grab;
      display: grid;
      place-items: center;
      color: white;
      font-weight: 900;
      font-size: 14px;
    }
    #error {
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 6;
      max-width: 420px;
      background: rgba(127, 29, 29, 0.96);
      color: white;
      padding: 10px 12px;
      border-radius: 12px;
      display: none;
      white-space: pre-wrap;
    }
    #mapMessage {
      position: absolute;
      left: 14px;
      bottom: 14px;
      z-index: 6;
      background: rgba(15, 23, 42, 0.82);
      color: white;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 12px;
      white-space: pre-wrap;
      max-width: min(460px, calc(100vw - 40px));
      pointer-events: none;
    }
    @media (max-width: 1000px) {
      #app {
        grid-template-columns: 1fr;
        grid-template-rows: minmax(320px, 44vh) 1fr;
      }
      #sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--panel-border);
      }
    }
  </style>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <section class="panel">
        <div id="modeLabel">2点合わせ</div>
        <div id="stepText">PDF overlay上の点をクリックしてください。</div>
      </section>

      <section class="panel">
        <h2>Frame</h2>
        <div class="field">
          <label for="frameOrder">順序</label>
          <select id="frameOrder">
            <option value="priority">優先度順</option>
            <option value="page">ページ順</option>
          </select>
        </div>
        <div class="field">
          <label for="frameSelect">対象 frame</label>
          <select id="frameSelect"></select>
        </div>
        <div id="frameInfo"></div>
      </section>

      <section class="panel">
        <h2>表示</h2>
        <div class="field">
          <label for="displayMode">Overlay 表示</label>
          <select id="displayMode">
            <option value="both">Both</option>
            <option value="map">Map</option>
            <option value="redlines">RedLines</option>
          </select>
        </div>
        <div class="field">
          <label for="opacityRange">Opacity</label>
          <input id="opacityRange" type="range" min="0.05" max="1" step="0.05" value="0.72" />
        </div>
      </section>

      <section class="panel">
        <h2>操作モード</h2>
        <div class="mode-grid">
          <button class="mode-button" data-mode="pair2">2点合わせ</button>
          <button class="mode-button" data-mode="pair4">4点合わせ</button>
          <button class="mode-button" data-mode="move">Move</button>
          <button class="mode-button" data-mode="fine">Fine-tune</button>
        </div>
        <div class="button-row" style="margin-top:10px;">
          <button id="undoPointButton" class="secondary">Undo last point</button>
          <button id="clearPointsButton" class="secondary">Clear points</button>
        </div>
      </section>

      <section class="panel">
        <h2>保存</h2>
        <div class="button-row">
          <button id="saveButton" class="primary">Save</button>
          <button id="copyJsonButton" class="secondary">Copy JSON</button>
        </div>
        <div class="button-row" style="margin-top:8px;">
          <button id="resetButton" class="secondary">Reset</button>
          <button id="nextButton" class="secondary">Next Frame</button>
        </div>
        <div id="statusText" style="margin-top:10px;"></div>
        <div id="saveHint" style="margin-top:8px;"></div>
      </section>

      <section class="panel">
        <h2>JSON Preview</h2>
        <pre id="jsonPreview"></pre>
      </section>

      <section class="panel">
        <h2>Hotkeys</h2>
        <div class="hint">矢印: 少し移動
Shift + 矢印: 大きく移動
Q / E: 回転
- / +: 縮小 / 拡大
R: Reset
S: Save
1 / 2 / 3: Map / RedLines / Both</div>
      </section>
    </aside>

    <main id="mapWrap">
      <div id="map"></div>
      <canvas id="overlayCanvas"></canvas>
      <div id="corner0" class="overlay-handle corner-handle" data-index="0"></div>
      <div id="corner1" class="overlay-handle corner-handle" data-index="1"></div>
      <div id="corner2" class="overlay-handle corner-handle" data-index="2"></div>
      <div id="corner3" class="overlay-handle corner-handle" data-index="3"></div>
      <div id="centerHandle" class="overlay-handle">+</div>
      <div id="error"></div>
      <div id="mapMessage"></div>
    </main>
  </div>

  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <script>
    window.__OVERLAY_GEOREF_CONFIG__ = __CONFIG_JSON__;
  </script>
  <script>
    const config = window.__OVERLAY_GEOREF_CONFIG__;
    const cornersOrder = ['top_left', 'top_right', 'bottom_right', 'bottom_left'];
    const MODE_META = {
      pair2: { label: '2点合わせ', requiredPairs: 2, transformLabel: 'similarity' },
      pair4: { label: '4点合わせ', requiredPairs: 4, transformLabel: 'projective' },
      move: { label: 'Move', requiredPairs: 0, transformLabel: 'move' },
      fine: { label: 'Fine-tune', requiredPairs: 0, transformLabel: 'fine' },
    };
    const imageCache = new Map();
    const frameState = new Map();
    const ui = {
      frameOrder: document.getElementById('frameOrder'),
      frameSelect: document.getElementById('frameSelect'),
      displayMode: document.getElementById('displayMode'),
      opacityRange: document.getElementById('opacityRange'),
      saveButton: document.getElementById('saveButton'),
      copyJsonButton: document.getElementById('copyJsonButton'),
      resetButton: document.getElementById('resetButton'),
      nextButton: document.getElementById('nextButton'),
      undoPointButton: document.getElementById('undoPointButton'),
      clearPointsButton: document.getElementById('clearPointsButton'),
      modeButtons: Array.from(document.querySelectorAll('.mode-button')),
      modeLabel: document.getElementById('modeLabel'),
      stepText: document.getElementById('stepText'),
      statusText: document.getElementById('statusText'),
      saveHint: document.getElementById('saveHint'),
      frameInfo: document.getElementById('frameInfo'),
      jsonPreview: document.getElementById('jsonPreview'),
      mapWrap: document.getElementById('mapWrap'),
      canvas: document.getElementById('overlayCanvas'),
      cornerHandles: [0, 1, 2, 3].map((index) => document.getElementById(`corner${index}`)),
      centerHandle: document.getElementById('centerHandle'),
      error: document.getElementById('error'),
      mapMessage: document.getElementById('mapMessage'),
    };

    const ctx = ui.canvas.getContext('2d');
    let currentFrameId = config.default_frame_id || (config.frames[0] && config.frames[0].frame_id) || '';
    let currentMode = 'pair2';
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

    function validateConfig() {
      if (!config || !Array.isArray(config.frames)) {
        showError('manifest 読み込みエラー: frames 配列がありません。');
        return false;
      }
      if (config.frames.length === 0) {
        showError('frame_manifest.json の frames が空です。build_overlay_georef_editor.py を再実行してください。');
        ui.statusText.textContent = 'frame count = 0';
        ui.stepText.textContent = 'frames が空のため表示できません。';
        return false;
      }
      return true;
    }

    function sortedFrames() {
      const frames = [...config.frames];
      if (ui.frameOrder.value === 'page') {
        frames.sort((a, b) => (a.page_no - b.page_no) || a.frame_id.localeCompare(b.frame_id));
      } else {
        frames.sort((a, b) => {
          const diff = Number(b.priority_score || 0) - Number(a.priority_score || 0);
          return diff || ((a.page_no - b.page_no) || a.frame_id.localeCompare(b.frame_id));
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
        option.textContent = `${frame.page_no} / ${frame.frame_id} / score=${Number(frame.priority_score || 0).toFixed(1)}`;
        ui.frameSelect.appendChild(option);
      }
      const preferred = (
        (selected && frames.some((frame) => frame.frame_id === selected) && selected)
        || (config.default_frame_id && frames.some((frame) => frame.frame_id === config.default_frame_id) && config.default_frame_id)
        || (frames[0] && frames[0].frame_id)
        || ''
      );
      ui.frameSelect.value = preferred;
      currentFrameId = preferred;
    }

    function cornersFromSaved(savedCorners) {
      return cornersOrder.map((name) => savedCorners[name].slice());
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

    function cornersFromView(frame) {
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

    function initialFrameCorners(frame) {
      if (frame.saved_corners) {
        return cornersFromSaved(frame.saved_corners);
      }
      if (frame.initial_corners) {
        return cornersFromSaved(frame.initial_corners);
      }
      return cornersFromView(frame);
    }

    function getState(frame) {
      if (!frameState.has(frame.frame_id)) {
        frameState.set(frame.frame_id, {
          corners: initialFrameCorners(frame),
          dirty: false,
          saveState: frame.has_saved_georef ? 'saved' : 'unsaved',
          alignmentPairs: [],
          lastAppliedMode: frame.saved_corners ? 'saved_georef' : (frame.initial_transform_source || 'fallback_view'),
          saveNotice: frame.has_saved_georef ? '既存保存JSONを読込済み' : '',
        });
      }
      return frameState.get(frame.frame_id);
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

    function drawPolygonOutline(screenCorners) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(screenCorners[0].x, screenCorners[0].y);
      for (let i = 1; i < screenCorners.length; i += 1) {
        ctx.lineTo(screenCorners[i].x, screenCorners[i].y);
      }
      ctx.closePath();
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'rgba(185,28,28,0.96)';
      ctx.setLineDash([12, 8]);
      ctx.stroke();
      ctx.restore();
    }

    function solveLinearSystem(matrix, values) {
      const n = matrix.length;
      const a = matrix.map((row, index) => [...row, values[index]]);
      for (let col = 0; col < n; col += 1) {
        let pivot = col;
        for (let row = col + 1; row < n; row += 1) {
          if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
            pivot = row;
          }
        }
        if (Math.abs(a[pivot][col]) < 1e-9) {
          return null;
        }
        if (pivot !== col) {
          [a[pivot], a[col]] = [a[col], a[pivot]];
        }
        const divisor = a[col][col];
        for (let k = col; k <= n; k += 1) {
          a[col][k] /= divisor;
        }
        for (let row = 0; row < n; row += 1) {
          if (row === col) {
            continue;
          }
          const factor = a[row][col];
          for (let k = col; k <= n; k += 1) {
            a[row][k] -= factor * a[col][k];
          }
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
      if (!params) {
        return null;
      }
      const [h11, h12, h13, h21, h22, h23, h31, h32] = params;
      const [x, y] = point;
      const denom = (h31 * x) + (h32 * y) + 1;
      if (Math.abs(denom) < 1e-9) {
        return null;
      }
      return [
        ((h11 * x) + (h12 * y) + h13) / denom,
        ((h21 * x) + (h22 * y) + h23) / denom,
      ];
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

    function drawFramedImage(image, opacity = 1) {
      const rect = ui.canvas.getBoundingClientRect();
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      const scale = Math.min(rect.width / width, rect.height / height);
      const drawWidth = width * scale;
      const drawHeight = height * scale;
      const x = (rect.width - drawWidth) / 2;
      const y = (rect.height - drawHeight) / 2;
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.drawImage(image, x, y, drawWidth, drawHeight);
      ctx.restore();
      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(185,28,28,0.9)';
      ctx.setLineDash([10, 6]);
      ctx.strokeRect(x, y, drawWidth, drawHeight);
      ctx.restore();
    }

    async function renderStaticPreview(frame, state) {
      resizeCanvas();
      ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
      const rect = ui.canvas.getBoundingClientRect();
      ctx.save();
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, rect.width, rect.height);
      ctx.restore();

      try {
        if (!frame.image_path) {
          throw new Error(`image_path が空です: ${frame.frame_id}`);
        }
        const displayMode = ui.displayMode.value;
        if (displayMode === 'map' || displayMode === 'both') {
          const image = await loadImage(frame.image_path);
          drawFramedImage(image, Number(ui.opacityRange.value || 0.72));
        }
        if (displayMode === 'redlines' || displayMode === 'both') {
          if (!frame.redlines_path) {
            throw new Error(`redlines_path が空です: ${frame.frame_id}`);
          }
          const image = await loadImage(frame.redlines_path);
          drawFramedImage(image, displayMode === 'both' ? 0.95 : Number(ui.opacityRange.value || 0.72));
        }
      } catch (error) {
        showError(error.message);
      }
      updateSidebar(frame, state);
      ui.mapMessage.textContent = [
        `page ${frame.page_no} / ${frame.frame_id}`,
        'Mapbox が未初期化のため、静的プレビューのみ表示しています。',
      ].join('\\n');
    }

    function pointerPosition(event) {
      const rect = ui.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function screenToLonLat(point) {
      const lngLat = map.unproject([point.x, point.y]);
      return [lngLat.lng, lngLat.lat];
    }

    function lonLatToLocal(point, ref) {
      const refLatRad = (ref[1] * Math.PI) / 180;
      return [
        6378137.0 * ((point[0] - ref[0]) * Math.PI / 180) * Math.cos(refLatRad),
        6378137.0 * ((point[1] - ref[1]) * Math.PI / 180),
      ];
    }

    function localToLonLat(point, ref) {
      const refLatRad = (ref[1] * Math.PI) / 180;
      return [
        ref[0] + ((point[0] / (6378137.0 * Math.cos(refLatRad))) * 180 / Math.PI),
        ref[1] + ((point[1] / 6378137.0) * 180 / Math.PI),
      ];
    }

    function fitSimilarityFromTwoPairs(pairs) {
      const s1 = [pairs[0].pdf.x, pairs[0].pdf.y];
      const s2 = [pairs[1].pdf.x, pairs[1].pdf.y];
      const ref = [
        (pairs[0].map[0] + pairs[1].map[0]) / 2,
        (pairs[0].map[1] + pairs[1].map[1]) / 2,
      ];
      const t1 = lonLatToLocal(pairs[0].map, ref);
      const t2 = lonLatToLocal(pairs[1].map, ref);
      const sv = [s2[0] - s1[0], s2[1] - s1[1]];
      const tv = [t2[0] - t1[0], t2[1] - t1[1]];
      const sNorm = Math.hypot(sv[0], sv[1]);
      const tNorm = Math.hypot(tv[0], tv[1]);
      if (sNorm < 1e-6 || tNorm < 1e-6) {
        throw new Error('2点が近すぎるため similarity を計算できません。');
      }
      const scale = tNorm / sNorm;
      const angle = Math.atan2(tv[1], tv[0]) - Math.atan2(sv[1], sv[0]);
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const srcCenter = [(s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2];
      const tgtCenter = [(t1[0] + t2[0]) / 2, (t1[1] + t2[1]) / 2];
      const rotatedCenter = [
        scale * ((cosA * srcCenter[0]) - (sinA * srcCenter[1])),
        scale * ((sinA * srcCenter[0]) + (cosA * srcCenter[1])),
      ];
      const translation = [
        tgtCenter[0] - rotatedCenter[0],
        tgtCenter[1] - rotatedCenter[1],
      ];
      return {
        ref,
        apply(point) {
          const rx = scale * ((cosA * point[0]) - (sinA * point[1])) + translation[0];
          const ry = scale * ((sinA * point[0]) + (cosA * point[1])) + translation[1];
          return localToLonLat([rx, ry], ref);
        },
      };
    }

    function fitProjectiveFromFourPairs(pairs) {
      const ref = pairs.reduce((acc, pair) => [acc[0] + pair.map[0], acc[1] + pair.map[1]], [0, 0]).map((value) => value / pairs.length);
      const src = pairs.map((pair) => [pair.pdf.x, pair.pdf.y]);
      const dst = pairs.map((pair) => lonLatToLocal(pair.map, ref));
      const params = fitProjective(src, dst);
      if (!params) {
        throw new Error('4点から projective 変換を計算できません。');
      }
      return {
        ref,
        params,
        apply(point) {
          const result = applyProjectivePoint(params, point);
          if (!result) {
            throw new Error('projective 変換結果が不正です。');
          }
          return localToLonLat(result, ref);
        },
      };
    }

    function imageCornerPixels(frame) {
      return [
        [0, 0],
        [frame.image_width_px, 0],
        [frame.image_width_px, frame.image_height_px],
        [0, frame.image_height_px],
      ];
    }

    function cornersFromTransform(frame, transformer) {
      return imageCornerPixels(frame).map((point) => transformer.apply(point));
    }

    function getHomographies(frame, corners) {
      const dst = projectedCorners(corners).map((point) => [point.x, point.y]);
      const src = imageCornerPixels(frame);
      return {
        sourceToScreen: fitProjective(src, dst),
        screenToSource: fitProjective(dst, src),
      };
    }

    function drawMarker(point, label, color, fill = true) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(point.x, point.y, 11, 0, Math.PI * 2);
      ctx.fillStyle = fill ? color : 'rgba(255,255,255,0.92)';
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = fill ? 'white' : color;
      ctx.font = '700 12px ui-sans-serif, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(label), point.x, point.y + 0.5);
      ctx.restore();
    }

    function drawAlignmentMarkers(frame, state, sourceToScreen) {
      state.alignmentPairs.forEach((pair, index) => {
        if (pair.pdf) {
          const pdfScreen = applyProjectivePoint(sourceToScreen, [pair.pdf.x, pair.pdf.y]);
          if (pdfScreen) {
            drawMarker({ x: pdfScreen[0], y: pdfScreen[1] }, index + 1, 'rgba(37,99,235,0.98)', true);
          }
        }
        if (pair.map) {
          const mapScreen = map.project(pair.map);
          drawMarker({ x: mapScreen.x, y: mapScreen.y }, index + 1, 'rgba(245,158,11,0.98)', false);
        }
        if (pair.pdf && pair.map) {
          const pdfScreen = applyProjectivePoint(sourceToScreen, [pair.pdf.x, pair.pdf.y]);
          const mapScreen = map.project(pair.map);
          if (pdfScreen) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pdfScreen[0], pdfScreen[1]);
            ctx.lineTo(mapScreen.x, mapScreen.y);
            ctx.strokeStyle = 'rgba(15,23,42,0.45)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 5]);
            ctx.stroke();
            ctx.restore();
          }
        }
      });
    }

    function updateHandles(screenCorners) {
      const center = screenCorners.reduce((acc, point) => ({ x: acc.x + point.x / 4, y: acc.y + point.y / 4 }), { x: 0, y: 0 });
      ui.cornerHandles.forEach((handle, index) => {
        handle.style.left = `${screenCorners[index].x}px`;
        handle.style.top = `${screenCorners[index].y}px`;
        handle.classList.toggle('active', currentMode === 'fine');
      });
      ui.centerHandle.style.left = `${center.x}px`;
      ui.centerHandle.style.top = `${center.y}px`;
      ui.centerHandle.style.opacity = currentMode === 'move' ? '1' : '0.72';
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

    function updateSidebar(frame, state) {
      const meta = MODE_META[currentMode];
      const payload = buildSavePayload(frame, state.corners);
      ui.modeLabel.textContent = meta.label;
      ui.frameInfo.innerHTML = [
        `page=${frame.page_no} frame=${frame.frame_id}`,
        `priority=${Number(frame.priority_score || 0).toFixed(1)} routes=${frame.route_count}`,
        `initial=${frame.initial_transform_source || 'fallback_view'}`,
        frame.has_saved_georef ? '<span class="badge">saved georef あり</span>' : '<span class="badge warn">saved georef なし</span>',
      ].join('<br />');
      ui.statusText.innerHTML = [
        state.dirty ? '未保存の変更あり' : (state.saveState === 'saved' ? '保存済み' : '未保存'),
        `last=${state.lastAppliedMode || 'none'}`,
      ].join('<br />');
      ui.saveHint.textContent = `${state.saveNotice || '保存先: data/manual_image_georef/'}\nダウンロード名: ${frame.suggested_download_filename}`;
      ui.jsonPreview.textContent = JSON.stringify(payload, null, 2);
      ui.modeButtons.forEach((button) => button.classList.toggle('active', button.dataset.mode === currentMode));
      ui.stepText.textContent = buildStepText(frame, state);
      ui.mapMessage.textContent = [
        `${meta.label} / page ${frame.page_no} / ${frame.frame_id}`,
        currentMode === 'pair2' || currentMode === 'pair4'
          ? '青=PDF点  橙=Mapbox点  点を順に対応付けてください'
          : '中心ハンドルまたはキーボードで調整できます',
      ].join('\\n');
    }

    function completePairCount(state) {
      return state.alignmentPairs.filter((pair) => pair.pdf && pair.map).length;
    }

    function expectedPairSide(state) {
      const requiredPairs = MODE_META[currentMode].requiredPairs;
      if (!requiredPairs) {
        return null;
      }
      if (completePairCount(state) >= requiredPairs) {
        return 'done';
      }
      const last = state.alignmentPairs[state.alignmentPairs.length - 1];
      if (!last || (last.pdf && last.map)) {
        return 'pdf';
      }
      if (last.pdf && !last.map) {
        return 'map';
      }
      return 'pdf';
    }

    function buildStepText(frame, state) {
      const meta = MODE_META[currentMode];
      if (currentMode === 'move') {
        return '中央ハンドルをドラッグ、または矢印キー / Q / E / +/- で調整してください。';
      }
      if (currentMode === 'fine') {
        return '四隅ハンドルをドラッグして微調整してください。';
      }
      const expected = expectedPairSide(state);
      const requiredPairs = meta.requiredPairs;
      const completed = completePairCount(state);
      if (expected === 'done') {
        return `${requiredPairs}組の対応点から ${meta.transformLabel} 変換を適用済みです。\nUndo last point または Clear points でやり直せます。`;
      }
      const stepIndex = state.alignmentPairs.length * 2 + (expected === 'map' ? 0 : 1);
      if (expected === 'pdf') {
        const nextIndex = completed + (state.alignmentPairs.length > completed ? 1 : 1);
        return `${stepIndex}/${requiredPairs * 2}: PDF overlay上の点 ${nextIndex} をクリックしてください。`;
      }
      return `${stepIndex}/${requiredPairs * 2}: Mapbox上の対応点 ${completed + 1} をクリックしてください。`;
    }

    async function renderOverlay() {
      if (!map) {
        const frame = frameById(currentFrameId);
        if (!frame) {
          showError(`対象 frame が見つかりません: ${currentFrameId || '(empty)'}`);
          return;
        }
        const state = getState(frame);
        await renderStaticPreview(frame, state);
        return;
      }
      const frame = frameById(currentFrameId);
      if (!frame) {
        showError(`対象 frame が見つかりません: ${currentFrameId || '(empty)'}`);
        return;
      }
      const state = getState(frame);
      resizeCanvas();
      ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
      const screenCorners = projectedCorners(state.corners);
      const opacity = Number(ui.opacityRange.value || 0.72);
      const displayMode = ui.displayMode.value;
      try {
        if (displayMode === 'map' || displayMode === 'both') {
          if (!frame.image_path) {
            throw new Error(`image_path が空です: ${frame.frame_id}`);
          }
          const image = await loadImage(frame.image_path);
          drawWarpedImage(image, screenCorners, opacity);
        }
        if (displayMode === 'redlines' || displayMode === 'both') {
          if (!frame.redlines_path) {
            throw new Error(`redlines_path が空です: ${frame.frame_id}`);
          }
          const image = await loadImage(frame.redlines_path);
          drawWarpedImage(image, screenCorners, displayMode === 'both' ? Math.min(1, opacity + 0.18) : opacity);
        }
      } catch (error) {
        showError(error.message);
      }
      drawPolygonOutline(screenCorners);
      const homographies = getHomographies(frame, state.corners);
      drawAlignmentMarkers(frame, state, homographies.sourceToScreen);
      updateHandles(screenCorners);
      updateSidebar(frame, state);
    }

    function markDirty(frame, state, modeLabel = currentMode) {
      state.dirty = true;
      state.saveState = 'unsaved';
      state.lastAppliedMode = modeLabel;
    }

    function fitMapToCorners(corners, immediate = false) {
      const bounds = new mapboxgl.LngLatBounds();
      corners.forEach((corner) => bounds.extend(corner));
      map.fitBounds(bounds, { padding: 40, duration: immediate ? 0 : 300, maxZoom: 17 });
    }

    function resetFrameState(frame, state) {
      state.corners = initialFrameCorners(frame);
      state.dirty = false;
      state.saveState = frame.has_saved_georef ? 'saved' : 'unsaved';
      state.alignmentPairs = [];
      state.lastAppliedMode = frame.saved_corners ? 'saved_georef' : (frame.initial_transform_source || 'fallback_view');
      state.saveNotice = frame.has_saved_georef ? '既存保存JSONを読込済み' : '初期配置に戻しました';
      if (map) {
        fitMapToCorners(state.corners, true);
      }
      renderOverlay();
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

    function currentScreenCorners(frame, state) {
      return projectedCorners(state.corners);
    }

    function applyScreenTransform(transformer, label) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const screenCorners = currentScreenCorners(frame, state);
      state.corners = transformer(screenCorners);
      markDirty(frame, state, label);
      renderOverlay();
    }

    function registerPdfPoint(sourcePoint) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (expectedPairSide(state) !== 'pdf') {
        return;
      }
      state.alignmentPairs.push({
        pdf: {
          x: clamp(sourcePoint[0], 0, frame.image_width_px),
          y: clamp(sourcePoint[1], 0, frame.image_height_px),
        },
      });
      state.saveNotice = `PDF点 ${state.alignmentPairs.length} を選択しました。`;
      renderOverlay();
    }

    function registerMapPoint(lonlat) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (expectedPairSide(state) !== 'map') {
        return;
      }
      const pair = state.alignmentPairs[state.alignmentPairs.length - 1];
      pair.map = [Number(lonlat[0].toFixed(7)), Number(lonlat[1].toFixed(7))];
      const requiredPairs = MODE_META[currentMode].requiredPairs;
      if (completePairCount(state) >= requiredPairs) {
        try {
          const completedPairs = state.alignmentPairs.slice(0, requiredPairs);
          const transformer = currentMode === 'pair2'
            ? fitSimilarityFromTwoPairs(completedPairs)
            : fitProjectiveFromFourPairs(completedPairs);
          state.corners = cornersFromTransform(frame, transformer);
          markDirty(frame, state, MODE_META[currentMode].transformLabel);
          state.saveNotice = `${MODE_META[currentMode].label}を適用しました。必要なら Fine-tune で微修正してください。`;
          fitMapToCorners(state.corners, false);
        } catch (error) {
          showError(error.message);
        }
      } else {
        state.saveNotice = `Mapbox点 ${completePairCount(state)} を選択しました。`;
      }
      renderOverlay();
    }

    function undoLastPoint() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (!state.alignmentPairs.length) {
        return;
      }
      const last = state.alignmentPairs[state.alignmentPairs.length - 1];
      if (last.map) {
        delete last.map;
        state.saveNotice = '最後の Mapbox 点を取り消しました。';
      } else {
        state.alignmentPairs.pop();
        state.saveNotice = '最後の PDF 点を取り消しました。';
      }
      renderOverlay();
    }

    function clearPoints() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      state.alignmentPairs = [];
      state.saveNotice = '対応点をクリアしました。';
      renderOverlay();
    }

    function onSave() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const payload = buildSavePayload(frame, state.corners);
      downloadJson(frame.suggested_download_filename, payload);
      state.dirty = false;
      state.saveState = 'saved';
      state.saveNotice = `保存用JSONをダウンロードしました。\n配置先: data/manual_image_georef/\nファイル名: ${frame.suggested_download_filename}`;
      renderOverlay();
    }

    async function onCopyJson() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const payload = JSON.stringify(buildSavePayload(frame, state.corners), null, 2);
      if (!navigator.clipboard) {
        showError('Clipboard API が利用できません。');
        return;
      }
      try {
        await navigator.clipboard.writeText(payload);
        state.saveNotice = 'JSON をクリップボードへコピーしました。';
        renderOverlay();
      } catch (error) {
        showError(`コピーに失敗しました: ${error.message}`);
      }
    }

    function onFrameChange() {
      currentFrameId = ui.frameSelect.value;
      const frame = frameById(currentFrameId);
      if (frame) {
        const state = getState(frame);
        if (map) {
          fitMapToCorners(state.corners, true);
        }
        renderOverlay();
      } else {
        showError(`frame select の値が不正です: ${currentFrameId || '(empty)'}`);
      }
    }

    function onNextFrame() {
      const frames = sortedFrames();
      const index = frames.findIndex((frame) => frame.frame_id === currentFrameId);
      const next = frames[(index + 1) % frames.length];
      if (!next) {
        return;
      }
      currentFrameId = next.frame_id;
      ui.frameSelect.value = next.frame_id;
      onFrameChange();
    }

    function setMode(mode) {
      currentMode = mode;
      renderOverlay();
    }

    function onMapWrapClick(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (!MODE_META[currentMode].requiredPairs || expectedPairSide(state) !== 'pdf') {
        return;
      }
      const pos = pointerPosition(event);
      const screenCorners = currentScreenCorners(frame, state);
      if (!pointInPolygon(pos, screenCorners)) {
        return;
      }
      const homographies = getHomographies(frame, state.corners);
      const sourcePoint = applyProjectivePoint(homographies.screenToSource, [pos.x, pos.y]);
      if (!sourcePoint) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      registerPdfPoint(sourcePoint);
    }

    function onMapClick(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (!MODE_META[currentMode].requiredPairs || expectedPairSide(state) !== 'map') {
        return;
      }
      registerMapPoint([event.lngLat.lng, event.lngLat.lat]);
    }

    function beginMoveDrag(event) {
      if (currentMode !== 'move') {
        return;
      }
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      event.preventDefault();
      event.stopPropagation();
      pointerState = {
        kind: 'move',
        start: pointerPosition(event),
        baseScreenCorners: currentScreenCorners(frame, state),
      };
      map.dragPan.disable();
    }

    function beginFineDrag(event, cornerIndex) {
      if (currentMode !== 'fine') {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      pointerState = { kind: 'fine', cornerIndex };
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
        markDirty(frame, state, 'move');
      } else if (pointerState.kind === 'fine') {
        state.corners[pointerState.cornerIndex] = screenToLonLat(pos);
        markDirty(frame, state, 'fine');
      }
      renderOverlay();
    }

    function onPointerUp() {
      if (!pointerState) {
        return;
      }
      pointerState = null;
      map.dragPan.enable();
    }

    function onKeyDown(event) {
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      const tagName = (event.target && event.target.tagName || '').toLowerCase();
      if (['input', 'select', 'textarea'].includes(tagName)) {
        return;
      }
      const moveStep = event.shiftKey ? 22 : 6;
      const rotateStep = (event.shiftKey ? 4 : 1) * Math.PI / 180;
      const scaleFactor = event.shiftKey ? 1.08 : 1.025;
      const key = event.key;
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const screenCorners = currentScreenCorners(frame, state);
      const center = screenCorners.reduce((acc, point) => ({ x: acc.x + point.x / 4, y: acc.y + point.y / 4 }), { x: 0, y: 0 });

      if (key === 'ArrowLeft') {
        event.preventDefault();
        applyScreenTransform((corners) => translateCorners(corners, -moveStep, 0), 'move');
      } else if (key === 'ArrowRight') {
        event.preventDefault();
        applyScreenTransform((corners) => translateCorners(corners, moveStep, 0), 'move');
      } else if (key === 'ArrowUp') {
        event.preventDefault();
        applyScreenTransform((corners) => translateCorners(corners, 0, -moveStep), 'move');
      } else if (key === 'ArrowDown') {
        event.preventDefault();
        applyScreenTransform((corners) => translateCorners(corners, 0, moveStep), 'move');
      } else if (key === 'q' || key === 'Q') {
        event.preventDefault();
        applyScreenTransform((corners) => rotateCorners(corners, center, -rotateStep), 'rotate');
      } else if (key === 'e' || key === 'E') {
        event.preventDefault();
        applyScreenTransform((corners) => rotateCorners(corners, center, rotateStep), 'rotate');
      } else if (key === '-' || key === '_') {
        event.preventDefault();
        applyScreenTransform((corners) => scaleCorners(corners, center, 1 / scaleFactor), 'scale');
      } else if (key === '+' || key === '=' ) {
        event.preventDefault();
        applyScreenTransform((corners) => scaleCorners(corners, center, scaleFactor), 'scale');
      } else if (key === 'r' || key === 'R') {
        event.preventDefault();
        resetFrameState(frame, state);
      } else if (key === 's' || key === 'S') {
        event.preventDefault();
        onSave();
      } else if (key === '1') {
        event.preventDefault();
        ui.displayMode.value = 'map';
        renderOverlay();
      } else if (key === '2') {
        event.preventDefault();
        ui.displayMode.value = 'redlines';
        renderOverlay();
      } else if (key === '3') {
        event.preventDefault();
        ui.displayMode.value = 'both';
        renderOverlay();
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
      ui.displayMode.addEventListener('change', renderOverlay);
      ui.opacityRange.addEventListener('input', renderOverlay);
      ui.saveButton.addEventListener('click', onSave);
      ui.copyJsonButton.addEventListener('click', onCopyJson);
      ui.resetButton.addEventListener('click', () => {
        const frame = frameById(currentFrameId);
        resetFrameState(frame, getState(frame));
      });
      ui.nextButton.addEventListener('click', onNextFrame);
      ui.undoPointButton.addEventListener('click', undoLastPoint);
      ui.clearPointsButton.addEventListener('click', clearPoints);
      ui.modeButtons.forEach((button) => {
        button.addEventListener('click', () => setMode(button.dataset.mode));
      });
      ui.mapWrap.addEventListener('click', onMapWrapClick, true);
      ui.centerHandle.addEventListener('pointerdown', beginMoveDrag);
      ui.cornerHandles.forEach((handle, index) => {
        handle.addEventListener('pointerdown', (event) => beginFineDrag(event, index));
      });
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
      window.addEventListener('keydown', onKeyDown);
    }

    function initMap() {
      if (!config.mapbox_access_token) {
        showError('MAPBOX_ACCESS_TOKEN がありません。.env を確認して build を再実行してください。');
        return;
      }
      if (typeof mapboxgl === 'undefined') {
        showError('Mapbox GL JS の読み込みに失敗しました。右ペインでは地図を表示できません。');
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
      map.on('error', (event) => {
        const message = event && event.error && event.error.message ? event.error.message : 'unknown map error';
        showError(`Mapbox エラー: ${message}`);
      });
      map.on('load', () => {
        clearError();
        const initial = frameById(currentFrameId) || config.frames[0];
        if (initial) {
          const state = getState(initial);
          fitMapToCorners(state.corners, true);
          renderOverlay();
        }
      });
      map.on('click', onMapClick);
      map.on('move', renderOverlay);
      map.on('resize', renderOverlay);
    }

    wireUi();
    if (validateConfig()) {
      repopulateFrameSelect();
      const initial = frameById(currentFrameId) || config.frames[0];
      if (initial) {
        renderStaticPreview(initial, getState(initial));
      } else {
        showError(`default frame を解決できません: ${config.default_frame_id || '(empty)'}`);
      }
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
