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
      --bg: #f5f3ee;
      --surface: #ffffff;
      --border: #d7d2c8;
      --text: #111827;
      --muted: #4b5563;
      --accent: #b91c1c;
      --accent-2: #1d4ed8;
      --accent-3: #f59e0b;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    #app {
      display: grid;
      grid-template-rows: auto auto 1fr;
      height: 100%;
      min-height: 0;
    }
    #toolbar {
      display: grid;
      grid-template-columns: auto minmax(260px, 340px) auto auto auto auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.92);
    }
    #statusbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.88);
      align-items: center;
    }
    select, button {
      font: inherit;
      min-height: 34px;
      border-radius: 10px;
      border: 1px solid rgba(15,23,42,0.14);
      background: white;
      padding: 6px 10px;
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: transparent; }
    .toolbarLabel {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      white-space: nowrap;
    }
    #frameCount {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(15,23,42,0.1);
      background: rgba(255,255,255,0.9);
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
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
      grid-template-columns: 45fr 55fr;
      min-height: 0;
    }
    .pane {
      position: relative;
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,244,239,0.98) 100%);
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
    .paneTitle {
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .paneControls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    #pdfCanvas {
      position: absolute;
      inset: 0;
      cursor: crosshair;
      touch-action: none;
    }
    #map {
      position: absolute;
      inset: 0;
    }
    .subtleButton {
      background: rgba(255,255,255,0.92);
    }
    #rightError {
      position: absolute;
      top: 58px;
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
      max-height: 38%;
      overflow: auto;
      background: rgba(15,23,42,0.9);
      color: #e2e8f0;
      border-radius: 12px;
      padding: 10px;
      font-size: 11px;
      line-height: 1.45;
      margin: 0;
    }
    #pdfError {
      position: absolute;
      left: 12px;
      right: 12px;
      bottom: 12px;
      z-index: 6;
      background: rgba(127,29,29,0.92);
      color: white;
      padding: 10px 12px;
      border-radius: 12px;
      display: none;
      white-space: pre-wrap;
      font-size: 12px;
    }
    details#advancedPanel {
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 6;
      width: min(320px, calc(100% - 24px));
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(15,23,42,0.12);
      border-radius: 12px;
      padding: 8px 10px;
      font-size: 12px;
    }
    details#advancedPanel summary {
      cursor: pointer;
      font-weight: 700;
    }
    .advancedRow {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 8px;
    }
    .advancedRow select {
      min-width: 160px;
    }
    .advancedNote {
      margin-top: 6px;
      color: var(--muted);
      line-height: 1.45;
    }
    @media (max-width: 1200px) {
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
      <div class="toolbarLabel">Frame</div>
      <select id="frameSelect"></select>
      <div id="frameCount"></div>
      <button id="undoButton">Undo</button>
      <button id="clearButton">Clear</button>
      <button id="previewButton">Preview</button>
      <button id="saveButton" class="primary">Save</button>
      <button id="nextButton">Next Frame</button>
      <div></div>
    </div>
    <div id="statusbar">
      <div id="stepText">次: 左の PDF をクリックしてください</div>
      <div id="metrics"></div>
      <div id="notice"></div>
    </div>
    <div id="workspace">
      <section class="pane">
        <div class="paneHeader">
          <div class="paneTitle">PDF</div>
          <div class="paneControls">
            <select id="pdfDisplaySelect">
              <option value="both">両方</option>
              <option value="map">元地図</option>
              <option value="redlines">赤線のみ</option>
            </select>
            <button id="fitPdfButton" class="subtleButton">Fit PDF</button>
          </div>
        </div>
        <canvas id="pdfCanvas"></canvas>
        <div id="pdfError"></div>
      </section>
      <section class="pane">
        <div class="paneHeader">
          <div class="paneTitle">Mapbox</div>
          <div class="paneControls">
            <span class="toolbarLabel">既定: 2点 similarity</span>
          </div>
        </div>
        <div id="map"></div>
        <div id="rightError"></div>
        <details id="advancedPanel">
          <summary>Advanced</summary>
          <div class="advancedRow">
            <label for="transformSelect">Preview transform</label>
            <select id="transformSelect">
              <option value="similarity">Similarity</option>
              <option value="projective">Projective (4点以上)</option>
            </select>
          </div>
          <div class="advancedNote">画像overlayのドラッグ移動・回転・四隅ハンドルは一旦外しています。必要ならここに再追加します。</div>
        </details>
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
    const PDF_INITIAL_FILL_RATIO = 0.82;
    const ui = {
      frameSelect: document.getElementById('frameSelect'),
      frameCount: document.getElementById('frameCount'),
      undoButton: document.getElementById('undoButton'),
      clearButton: document.getElementById('clearButton'),
      previewButton: document.getElementById('previewButton'),
      saveButton: document.getElementById('saveButton'),
      nextButton: document.getElementById('nextButton'),
      pdfDisplaySelect: document.getElementById('pdfDisplaySelect'),
      fitPdfButton: document.getElementById('fitPdfButton'),
      transformSelect: document.getElementById('transformSelect'),
      stepText: document.getElementById('stepText'),
      metrics: document.getElementById('metrics'),
      notice: document.getElementById('notice'),
      pdfCanvas: document.getElementById('pdfCanvas'),
      pdfError: document.getElementById('pdfError'),
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

    function currentTransformChoice() {
      return ui.transformSelect.value;
    }

    function clearPdfError() {
      ui.pdfError.style.display = 'none';
      ui.pdfError.textContent = '';
    }

    function showPdfError(message) {
      ui.pdfError.style.display = 'block';
      ui.pdfError.textContent = message;
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

    function getState(frame) {
      if (!frameState.has(frame.frame_id)) {
        frameState.set(frame.frame_id, {
          controlPoints: [],
          preview: null,
          dirty: false,
          notice: frame.has_saved_georef ? '既存 saved georef あり' : '',
          pdfView: {
            zoom: 1,
            panX: 0,
            panY: 0,
            fitApplied: false,
          },
          dragState: null,
        });
      }
      return frameState.get(frame.frame_id);
    }

    function completedPairs(state) {
      return state.controlPoints.filter((pair) => pair.pdf_px && pair.lonlat).length;
    }

    function expectedSide(state) {
      const last = state.controlPoints[state.controlPoints.length - 1];
      if (!last || (last.pdf_px && last.lonlat)) return 'pdf';
      return 'map';
    }

    function populateFrameSelect() {
      const frames = sortedFrames();
      ui.frameSelect.innerHTML = '';
      frames.forEach((frame, index) => {
        const option = document.createElement('option');
        option.value = frame.frame_id;
        option.textContent = `${index + 1}/${frames.length}  ${frame.page_no} / ${frame.frame_id}`;
        ui.frameSelect.appendChild(option);
      });
      currentFrameId = config.default_frame_id || frames[0].frame_id;
      ui.frameSelect.value = currentFrameId;
      ui.frameCount.textContent = `${frames.length} 件`;
    }

    function updateStepText(frame, state) {
      const side = expectedSide(state);
      const done = completedPairs(state);
      const nextIndex = done + 1;
      if (side === 'pdf') {
        ui.stepText.textContent = `次: 左の PDF をクリックしてください (${nextIndex}点目)`;
      } else {
        ui.stepText.textContent = `次: 右の Mapbox をクリックしてください (${nextIndex}点目)`;
      }

      const preview = state.preview;
      const modelBits = [];
      if (preview && preview.metrics_by_model) {
        if (Number.isFinite(preview.metrics_by_model.similarity)) {
          modelBits.push(`sim ${preview.metrics_by_model.similarity.toFixed(2)}m`);
        }
        if (Number.isFinite(preview.metrics_by_model.affine)) {
          modelBits.push(`aff ${preview.metrics_by_model.affine.toFixed(2)}m`);
        }
        if (Number.isFinite(preview.metrics_by_model.projective)) {
          modelBits.push(`proj ${preview.metrics_by_model.projective.toFixed(2)}m`);
        }
      }
      const activeRmse = preview && Number.isFinite(preview.rmse_m) ? `${preview.rmse_m.toFixed(2)} m` : '-';
      ui.metrics.innerHTML = `<strong>frame</strong>: ${frame.frame_id} / <strong>pairs</strong>: ${done} / <strong>active</strong>: ${preview ? preview.transform_type : 'none'} / <strong>RMSE</strong>: ${activeRmse}${modelBits.length ? ` / <strong>compare</strong>: ${modelBits.join(' | ')}` : ''}`;
      ui.notice.textContent = state.notice || '';
    }

    async function loadImage(url) {
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

    function fitPdfView(frame, rect, state) {
      const widthScale = rect.width / frame.image_width_px;
      const targetHeightScale = (rect.height * PDF_INITIAL_FILL_RATIO) / frame.image_height_px;
      const baseScale = Math.max(widthScale, targetHeightScale);
      const scaledWidth = frame.image_width_px * baseScale;
      const scaledHeight = frame.image_height_px * baseScale;
      const baseX = (rect.width - scaledWidth) / 2;
      const baseY = (rect.height - scaledHeight) / 2;
      state.pdfView.zoom = 1;
      state.pdfView.panX = baseX;
      state.pdfView.panY = baseY;
      state.pdfView.fitApplied = true;
    }

    function currentPdfLayout(frame, rect, state) {
      if (!state.pdfView.fitApplied) fitPdfView(frame, rect, state);
      const widthScale = rect.width / frame.image_width_px;
      const targetHeightScale = (rect.height * PDF_INITIAL_FILL_RATIO) / frame.image_height_px;
      const baseScale = Math.max(widthScale, targetHeightScale);
      const scale = baseScale * state.pdfView.zoom;
      const width = frame.image_width_px * scale;
      const height = frame.image_height_px * scale;
      return {
        x: state.pdfView.panX,
        y: state.pdfView.panY,
        scale,
        width,
        height,
      };
    }

    function canvasToPdf(layout, x, y) {
      return [
        (x - layout.x) / layout.scale,
        (y - layout.y) / layout.scale,
      ];
    }

    function pdfToCanvas(layout, point) {
      return {
        x: layout.x + (point[0] * layout.scale),
        y: layout.y + (point[1] * layout.scale),
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

    async function drawPdfPane() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const rect = resizePdfCanvas();
      clearPdfError();
      pdfCtx.clearRect(0, 0, rect.width, rect.height);
      pdfCtx.fillStyle = '#ebe7df';
      pdfCtx.fillRect(0, 0, rect.width, rect.height);
      const layout = currentPdfLayout(frame, rect, state);

      try {
        const mode = ui.pdfDisplaySelect.value;
        if (mode !== 'redlines') {
          const baseImage = await loadImage(frame.image_path);
          pdfCtx.drawImage(baseImage, layout.x, layout.y, layout.width, layout.height);
        }
        if (mode !== 'map') {
          const redImage = await loadImage(frame.redlines_path);
          pdfCtx.globalAlpha = mode === 'both' ? 0.92 : 1;
          pdfCtx.drawImage(redImage, layout.x, layout.y, layout.width, layout.height);
          pdfCtx.globalAlpha = 1;
        }
      } catch (error) {
        showPdfError(error.message);
      }

      pdfCtx.strokeStyle = 'rgba(29,78,216,0.95)';
      pdfCtx.lineWidth = 3;
      pdfCtx.strokeRect(layout.x, layout.y, layout.width, layout.height);

      state.controlPoints.forEach((pair, index) => {
        if (!pair.pdf_px) return;
        const point = pdfToCanvas(layout, pair.pdf_px);
        drawMarker(pdfCtx, point.x, point.y, index + 1, 'rgba(29,78,216,0.95)', '#ffffff', '#ffffff');
      });
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

    function solveLeastSquares(rows, values, columns) {
      const ata = Array.from({ length: columns }, () => Array(columns).fill(0));
      const atb = Array(columns).fill(0);
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex];
        for (let i = 0; i < columns; i += 1) {
          atb[i] += row[i] * values[rowIndex];
          for (let j = 0; j < columns; j += 1) {
            ata[i][j] += row[i] * row[j];
          }
        }
      }
      return solveLinearSystem(ata, atb);
    }

    function fitSimilarityModel(src, dst, refLonLat) {
      const rows = [];
      const values = [];
      for (let index = 0; index < src.length; index += 1) {
        const [x, y] = src[index];
        const [u, v] = dst[index];
        rows.push([x, -y, 1, 0]);
        values.push(u);
        rows.push([y, x, 0, 1]);
        values.push(v);
      }
      const params = solveLeastSquares(rows, values, 4);
      if (!params) throw new Error('similarity 変換を計算できません。');
      const [a, b, tx, ty] = params;
      return {
        type: 'similarity',
        refLonLat,
        applyLocal(point) {
          const [x, y] = point;
          return [
            (a * x) - (b * y) + tx,
            (b * x) + (a * y) + ty,
          ];
        },
        apply(point) {
          return metersToLonLat(this.applyLocal(point), refLonLat);
        },
      };
    }

    function fitAffineModel(src, dst, refLonLat) {
      const rows = [];
      const values = [];
      for (let index = 0; index < src.length; index += 1) {
        const [x, y] = src[index];
        const [u, v] = dst[index];
        rows.push([x, y, 1, 0, 0, 0]);
        values.push(u);
        rows.push([0, 0, 0, x, y, 1]);
        values.push(v);
      }
      const params = solveLeastSquares(rows, values, 6);
      if (!params) throw new Error('affine 変換を計算できません。');
      const [a, b, c, d, e, f] = params;
      return {
        type: 'affine',
        refLonLat,
        applyLocal(point) {
          const [x, y] = point;
          return [
            (a * x) + (b * y) + c,
            (d * x) + (e * y) + f,
          ];
        },
        apply(point) {
          return metersToLonLat(this.applyLocal(point), refLonLat);
        },
      };
    }

    function fitProjectiveParameters(src, dst) {
      const rows = [];
      const values = [];
      for (let index = 0; index < src.length; index += 1) {
        const [x, y] = src[index];
        const [u, v] = dst[index];
        rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
        values.push(u);
        rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
        values.push(v);
      }
      return solveLeastSquares(rows, values, 8);
    }

    function fitProjectiveModel(src, dst, refLonLat) {
      const params = fitProjectiveParameters(src, dst);
      if (!params) throw new Error('projective 変換を計算できません。');
      return {
        type: 'projective',
        refLonLat,
        applyLocal(point) {
          const [h11, h12, h13, h21, h22, h23, h31, h32] = params;
          const [x, y] = point;
          const denom = (h31 * x) + (h32 * y) + 1;
          if (Math.abs(denom) < 1e-9) throw new Error('projective 変換結果が不正です。');
          return [
            ((h11 * x) + (h12 * y) + h13) / denom,
            ((h21 * x) + (h22 * y) + h23) / denom,
          ];
        },
        apply(point) {
          return metersToLonLat(this.applyLocal(point), refLonLat);
        },
      };
    }

    function rmseForModel(model, pairs) {
      let sumSq = 0;
      for (const pair of pairs) {
        const predicted = model.applyLocal(pair.pdf_px);
        const target = localMeters(pair.lonlat, model.refLonLat);
        const error = Math.hypot(predicted[0] - target[0], predicted[1] - target[1]);
        sumSq += error * error;
      }
      return Math.sqrt(sumSq / pairs.length);
    }

    function buildModels(pairs) {
      const refLonLat = pairs
        .reduce((acc, pair) => [acc[0] + pair.lonlat[0], acc[1] + pair.lonlat[1]], [0, 0])
        .map((value) => value / pairs.length);
      const src = pairs.map((pair) => pair.pdf_px);
      const dst = pairs.map((pair) => localMeters(pair.lonlat, refLonLat));
      const models = {};
      if (pairs.length >= 2) models.similarity = fitSimilarityModel(src, dst, refLonLat);
      if (pairs.length >= 3) models.affine = fitAffineModel(src, dst, refLonLat);
      if (pairs.length >= 4) models.projective = fitProjectiveModel(src, dst, refLonLat);
      return models;
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

    function transformRouteGeometry(frame, geometry, model) {
      const transformLine = (line) => line.map((coord) => model.apply(routeCoordToPdfPx(frame, coord)).map((value) => Number(value.toFixed(7))));
      if (geometry.type === 'LineString') {
        return { type: 'LineString', coordinates: transformLine(geometry.coordinates) };
      }
      return { type: 'MultiLineString', coordinates: geometry.coordinates.map(transformLine) };
    }

    function buildPreview(frame, state) {
      const pairs = state.controlPoints.filter((pair) => pair.pdf_px && pair.lonlat);
      if (pairs.length < 2) {
        throw new Error('2組以上の対応点が必要です。');
      }
      const models = buildModels(pairs);
      const selected = currentTransformChoice();
      let activeModel = models.similarity;
      if (selected === 'projective' && models.projective) {
        activeModel = models.projective;
      }

      const metricsByModel = {
        similarity: models.similarity ? rmseForModel(models.similarity, pairs) : null,
        affine: models.affine ? rmseForModel(models.affine, pairs) : null,
        projective: models.projective ? rmseForModel(models.projective, pairs) : null,
      };

      const transformedControl = [];
      const residuals = [];
      let sumSq = 0;
      pairs.forEach((pair, index) => {
        const predicted = activeModel.apply(pair.pdf_px);
        transformedControl.push({ label: String(index + 1), lonlat: predicted });
        const targetMeters = localMeters(pair.lonlat, activeModel.refLonLat);
        const predMeters = localMeters(predicted, activeModel.refLonLat);
        const error = Math.hypot(predMeters[0] - targetMeters[0], predMeters[1] - targetMeters[1]);
        sumSq += error * error;
        residuals.push({
          label: String(index + 1),
          coordinates: [predicted, pair.lonlat],
          error_m: error,
        });
      });
      const rmse = Math.sqrt(sumSq / pairs.length);
      const corners = imageCorners(frame).map((point) => activeModel.apply(point).map((value) => Number(value.toFixed(7))));
      const transformedRoutes = {
        type: 'FeatureCollection',
        features: (frame.route_geojson?.features || []).map((feature) => ({
          type: 'Feature',
          geometry: transformRouteGeometry(frame, feature.geometry, activeModel),
          properties: { ...feature.properties },
        })),
      };
      return {
        transform_type: activeModel.type,
        rmse_m: rmse,
        metrics_by_model: metricsByModel,
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
      if (!state.preview) throw new Error('Preview を先に実行してください。');
      return {
        page_no: frame.page_no,
        frame_id: frame.frame_id,
        transform_type: state.preview.transform_type,
        control_points: state.controlPoints
          .filter((pair) => pair.pdf_px && pair.lonlat)
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
        corners_lonlat: state.preview.corners_lonlat,
        rmse_m: Number(state.preview.rmse_m.toFixed(3)),
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
          transform_type: currentTransformChoice(),
          control_points: state.controlPoints,
        }, null, 2);
      }
    }

    function coordsFromCorners(corners) {
      if (!corners) return [];
      return ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        .filter((key) => Array.isArray(corners[key]))
        .map((key) => corners[key]);
    }

    function fitMapToCoords(coords) {
      if (!mapReady || !coords.length) return false;
      if (coords.length === 1) {
        map.jumpTo({ center: coords[0], zoom: 14.5 });
        return true;
      }
      const bounds = new mapboxgl.LngLatBounds();
      coords.forEach((coord) => bounds.extend(coord));
      map.fitBounds(bounds, { padding: 48, duration: 0, maxZoom: 16.5 });
      return true;
    }

    function fitMapToFrameInitial(frame) {
      if (!mapReady) return;
      if (fitMapToCoords(coordsFromCorners(frame.saved_corners))) return;
      if (fitMapToCoords((frame.manual_geo_points || []).map((point) => [point.longitude, point.latitude]))) return;
      if (fitMapToCoords((frame.auto_geo_points || []).map((point) => [point.longitude, point.latitude]))) return;
      if (fitMapToCoords(coordsFromCorners(frame.initial_corners))) return;
      map.jumpTo({
        center: frame.initial_center || config.initial_map_center || [133.5, 33.8],
        zoom: frame.initial_zoom || config.initial_map_zoom || 8.5,
      });
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
        const compare = [];
        const metrics = state.preview.metrics_by_model || {};
        if (Number.isFinite(metrics.affine)) compare.push(`aff ${metrics.affine.toFixed(2)}m`);
        if (Number.isFinite(metrics.projective)) compare.push(`proj ${metrics.projective.toFixed(2)}m`);
        state.notice = `Preview 更新 / active ${state.preview.transform_type} / RMSE ${state.preview.rmse_m.toFixed(2)} m${compare.length ? ` / compare ${compare.join(' | ')}` : ''}`;
        updateUi();
        fitMapToCoords(coordsFromCorners(state.preview.corners_lonlat));
      } catch (error) {
        state.notice = error.message;
        updateUi();
      }
    }

    function resetFrame(frame, { keepNotice = false } = {}) {
      const state = getState(frame);
      state.controlPoints = [];
      state.preview = null;
      state.dirty = false;
      if (!keepNotice) state.notice = '';
      renderAll();
      fitMapToFrameInitial(frame);
    }

    function onPdfCanvasClick(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      if (state.dragState && state.dragState.moved) {
        state.dragState = null;
        return;
      }
      if (expectedSide(state) !== 'pdf') return;
      const rect = ui.pdfCanvas.getBoundingClientRect();
      const layout = currentPdfLayout(frame, rect, state);
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const point = canvasToPdf(layout, x, y);
      if (point[0] < 0 || point[0] > frame.image_width_px || point[1] < 0 || point[1] > frame.image_height_px) return;
      state.controlPoints.push({
        pdf_px: [Number(point[0].toFixed(3)), Number(point[1].toFixed(3))],
        lonlat: null,
      });
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
      if (completedPairs(state) >= 2) {
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
        if (!state.preview) state.preview = buildPreview(frame, state);
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
      const nextFrame = frameById(currentFrameId);
      const nextState = getState(nextFrame);
      nextState.pdfView.fitApplied = false;
      renderAll();
      fitMapToFrameInitial(nextFrame);
    }

    function onFrameChange() {
      currentFrameId = ui.frameSelect.value;
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      state.pdfView.fitApplied = false;
      renderAll();
      fitMapToFrameInitial(frame);
    }

    function onFitPdf() {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const rect = ui.pdfCanvas.getBoundingClientRect();
      fitPdfView(frame, rect, state);
      renderAll();
    }

    function beginPdfPan(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      state.dragState = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        panX: state.pdfView.panX,
        panY: state.pdfView.panY,
        moved: false,
      };
      ui.pdfCanvas.setPointerCapture(event.pointerId);
    }

    function movePdfPan(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const drag = state.dragState;
      if (!drag || drag.pointerId !== event.pointerId) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      state.pdfView.panX = drag.panX + dx;
      state.pdfView.panY = drag.panY + dy;
      renderAll();
    }

    function endPdfPan(event) {
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const drag = state.dragState;
      if (!drag || drag.pointerId !== event.pointerId) return;
      state.dragState = drag;
      ui.pdfCanvas.releasePointerCapture(event.pointerId);
      setTimeout(() => {
        if (state.dragState === drag) state.dragState = null;
      }, 0);
    }

    function onPdfWheel(event) {
      event.preventDefault();
      const frame = frameById(currentFrameId);
      const state = getState(frame);
      const rect = ui.pdfCanvas.getBoundingClientRect();
      const before = currentPdfLayout(frame, rect, state);
      const anchorX = event.clientX - rect.left;
      const anchorY = event.clientY - rect.top;
      const imagePoint = canvasToPdf(before, anchorX, anchorY);
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      state.pdfView.zoom = clamp(state.pdfView.zoom * factor, 0.6, 8);
      const after = currentPdfLayout(frame, rect, state);
      state.pdfView.panX += anchorX - (after.x + imagePoint[0] * after.scale);
      state.pdfView.panY += anchorY - (after.y + imagePoint[1] * after.scale);
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
          paint: { 'line-color': '#dc2626', 'line-width': 3.2 },
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
            'circle-radius': 7,
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
          paint: { 'text-color': '#ffffff' },
        });
        map.on('click', onMapClick);
        map.on('error', (event) => {
          const message = event && event.error && event.error.message ? event.error.message : 'Mapbox error';
          showRightError(message);
        });
        fitMapToFrameInitial(frameById(currentFrameId));
        renderAll();
      });
    }

    function wireUi() {
      ui.frameSelect.addEventListener('change', onFrameChange);
      ui.undoButton.addEventListener('click', onUndo);
      ui.clearButton.addEventListener('click', onClear);
      ui.previewButton.addEventListener('click', previewCurrent);
      ui.saveButton.addEventListener('click', onSave);
      ui.nextButton.addEventListener('click', onNextFrame);
      ui.pdfDisplaySelect.addEventListener('change', renderAll);
      ui.fitPdfButton.addEventListener('click', onFitPdf);
      ui.transformSelect.addEventListener('change', () => {
        const frame = frameById(currentFrameId);
        const state = getState(frame);
        if (completedPairs(state) >= 2) {
          previewCurrent();
        } else {
          updateUi();
        }
      });
      ui.pdfCanvas.addEventListener('click', onPdfCanvasClick);
      ui.pdfCanvas.addEventListener('pointerdown', beginPdfPan);
      ui.pdfCanvas.addEventListener('pointermove', movePdfPan);
      ui.pdfCanvas.addEventListener('pointerup', endPdfPan);
      ui.pdfCanvas.addEventListener('pointercancel', endPdfPan);
      ui.pdfCanvas.addEventListener('wheel', onPdfWheel, { passive: false });
      window.addEventListener('resize', () => {
        const frame = frameById(currentFrameId);
        const state = getState(frame);
        state.pdfView.fitApplied = false;
        renderAll();
        fitMapToFrameInitial(frame);
      });
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
                    "manual_geo_points": serializable_manual_geo_points(manual_gcp),
                    "auto_geo_points": serializable_auto_geo_points(auto_rows),
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
        "frame_count": len(manifest_frames),
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
