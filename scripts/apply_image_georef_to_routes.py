#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MAPBOX_SCALE = 0.01
SHIKOKU_LON_RANGE = (132.0, 135.5)
SHIKOKU_LAT_RANGE = (32.5, 34.8)


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
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    u = ((h11 * x) + (h12 * y) + h13) / denom
    v = ((h21 * x) + (h22 * y) + h23) / denom
    return np.column_stack([u, v])


def synthetic_to_pdf(coord: list[float]) -> tuple[float, float]:
    return (coord[0] / MAPBOX_SCALE, -coord[1] / MAPBOX_SCALE)


def frame_row_by_key(path: Path, page_no: int, frame_id: str) -> dict[str, Any]:
    for row in read_csv_rows(path):
        if int(row["page_no"]) == page_no and row["frame_id"] == frame_id:
            return row
    raise SystemExit(f"frame が見つかりません: page={page_no} frame={frame_id}")


def load_routes(path: Path, page_no: int, frame_id: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        feature for feature in data.get("features", [])
        if int(feature.get("properties", {}).get("page_no", -1)) == page_no
        and feature.get("properties", {}).get("frame_id") == frame_id
    ]


def frame_bbox_from_json_or_csv(georef_json: dict[str, Any], frame_row: dict[str, Any]) -> dict[str, float]:
    bbox = georef_json.get("pdf_frame_bbox") or georef_json.get("frame", {}).get("bbox_pt")
    if bbox:
        return {
            "x0": float(bbox["x0"]),
            "y0": float(bbox["y0"]),
            "x1": float(bbox["x1"]),
            "y1": float(bbox["y1"]),
        }
    return {
        "x0": float(frame_row["x0_pt"]),
        "y0": float(frame_row["y0_pt"]),
        "x1": float(frame_row["x1_pt"]),
        "y1": float(frame_row["y1_pt"]),
    }


def build_projective_params(georef_json: dict[str, Any], pdf_bbox: dict[str, float]) -> np.ndarray:
    corners = georef_json["corners_lonlat"]
    source = np.array(
        [
            [pdf_bbox["x0"], pdf_bbox["y0"]],
            [pdf_bbox["x1"], pdf_bbox["y0"]],
            [pdf_bbox["x1"], pdf_bbox["y1"]],
            [pdf_bbox["x0"], pdf_bbox["y1"]],
        ],
        dtype=float,
    )
    target = np.array(
        [
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ],
        dtype=float,
    )
    return fit_projective(source, target)


def point_in_shikoku(lon: float, lat: float) -> bool:
    return SHIKOKU_LON_RANGE[0] <= lon <= SHIKOKU_LON_RANGE[1] and SHIKOKU_LAT_RANGE[0] <= lat <= SHIKOKU_LAT_RANGE[1]


def transform_line(line: list[list[float]], params: np.ndarray) -> tuple[list[list[float]], list[str]]:
    pdf_points = np.array([synthetic_to_pdf(coord) for coord in line], dtype=float)
    lonlat = apply_projective(pdf_points, params)
    issues: list[str] = []
    coords: list[list[float]] = []
    for lon, lat in lonlat:
        if not np.isfinite(lon) or not np.isfinite(lat):
            issues.append("nonfinite_coordinate")
            continue
        if not point_in_shikoku(float(lon), float(lat)):
            issues.append("out_of_shikoku_bounds")
        coords.append([round(float(lon), 7), round(float(lat), 7)])
    return coords, sorted(set(issues))


def transform_geometry(geometry: dict[str, Any], params: np.ndarray) -> tuple[dict[str, Any] | None, list[str]]:
    if geometry["type"] == "LineString":
        coords, issues = transform_line(geometry["coordinates"], params)
        if len(coords) < 2:
            return None, sorted(set(issues + ["too_few_points"]))
        return {"type": "LineString", "coordinates": coords}, issues
    if geometry["type"] == "MultiLineString":
        lines: list[list[list[float]]] = []
        issues: list[str] = []
        for line in geometry["coordinates"]:
            coords, line_issues = transform_line(line, params)
            issues.extend(line_issues)
            if len(coords) >= 2:
                lines.append(coords)
        if not lines:
            return None, sorted(set(issues + ["too_few_points"]))
        return {"type": "MultiLineString", "coordinates": lines}, sorted(set(issues))
    return None, ["unsupported_geometry"]


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["route_id", "status", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def build_debug_html(
    *,
    token: str,
    transformed_geojson: dict[str, Any],
    georef_json: dict[str, Any],
    map_image_path: str | None,
    redlines_image_path: str | None,
) -> str:
    config = {
        "mapboxToken": token,
        "transformedRoutes": transformed_geojson,
        "georef": georef_json,
        "mapImagePath": map_image_path,
        "redlinesImagePath": redlines_image_path,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    template = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image Georef Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
  <style>
    html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { display: grid; grid-template-rows: auto 1fr; }
    #toolbar {
      display: flex;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(15, 23, 42, 0.12);
      background: rgba(255,255,255,0.96);
    }
    #map { min-height: 0; }
    #message {
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 5;
      background: rgba(15, 23, 42, 0.82);
      color: white;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div id="toolbar">
    <label><input id="toggleMapImage" type="checkbox" checked /> map画像</label>
    <label><input id="toggleRedlines" type="checkbox" /> redlines画像</label>
  </div>
  <div id="map"></div>
  <div id="message"></div>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
  <script>
    const config = __CONFIG_JSON__;
    const corners = config.georef.corners_lonlat;
    const cornerArray = [
      corners.top_left,
      corners.top_right,
      corners.bottom_right,
      corners.bottom_left,
    ];

    const message = document.getElementById('message');
    message.textContent = [
      `frame=${config.georef.frame_id}`,
      `page=${config.georef.page_no}`,
      `routes=${config.transformedRoutes.features.length}`,
      `generated=${config.generatedAt}`,
    ].join('\\n');

    if (config.mapboxToken) {
      mapboxgl.accessToken = config.mapboxToken;
    }

    const map = new mapboxgl.Map({
      container: 'map',
      style: 'mapbox://styles/mapbox/streets-v12',
      center: cornerArray[0],
      zoom: 13,
    });

    map.on('load', () => {
      map.addSource('routes', {
        type: 'geojson',
        data: config.transformedRoutes,
      });
      map.addLayer({
        id: 'routes-line',
        type: 'line',
        source: 'routes',
        paint: {
          'line-color': '#dc2626',
          'line-width': 3,
        },
      });

      map.addSource('frame', {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [{
            type: 'Feature',
            geometry: {
              type: 'Polygon',
              coordinates: [[
                corners.top_left,
                corners.top_right,
                corners.bottom_right,
                corners.bottom_left,
                corners.top_left,
              ]],
            },
          }],
        },
      });
      map.addLayer({
        id: 'frame-outline',
        type: 'line',
        source: 'frame',
        paint: {
          'line-color': '#2563eb',
          'line-width': 2,
          'line-dasharray': [2, 2],
        },
      });

      if (config.mapImagePath) {
        map.addSource('map-image', {
          type: 'image',
          url: config.mapImagePath,
          coordinates: cornerArray,
        });
        map.addLayer({
          id: 'map-image-layer',
          type: 'raster',
          source: 'map-image',
          paint: { 'raster-opacity': 0.55 },
        });
      }

      if (config.redlinesImagePath) {
        map.addSource('redlines-image', {
          type: 'image',
          url: config.redlinesImagePath,
          coordinates: cornerArray,
        });
        map.addLayer({
          id: 'redlines-image-layer',
          type: 'raster',
          source: 'redlines-image',
          layout: { visibility: 'none' },
          paint: { 'raster-opacity': 0.9 },
        });
      }

      const bounds = new mapboxgl.LngLatBounds();
      for (const point of cornerArray) {
        bounds.extend(point);
      }
      for (const feature of config.transformedRoutes.features) {
        const geometry = feature.geometry || {};
        const lines = geometry.type === 'LineString' ? [geometry.coordinates] : (geometry.coordinates || []);
        for (const line of lines) {
          for (const coord of line) {
            bounds.extend(coord);
          }
        }
      }
      map.fitBounds(bounds, { padding: 40, duration: 0 });
    });

    document.getElementById('toggleMapImage').addEventListener('change', (event) => {
      if (map.getLayer('map-image-layer')) {
        map.setLayoutProperty('map-image-layer', 'visibility', event.target.checked ? 'visible' : 'none');
      }
    });
    document.getElementById('toggleRedlines').addEventListener('change', (event) => {
      if (map.getLayer('redlines-image-layer')) {
        map.setLayoutProperty('redlines-image-layer', 'visibility', event.target.checked ? 'visible' : 'none');
      }
    });
  </script>
</body>
</html>
"""
    return template.replace("__CONFIG_JSON__", json.dumps(config, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply saved overlay image georeferencing to frame routes.")
    parser.add_argument("--georef-json", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--frames", type=Path, default=Path("artifacts/step2/frames.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--overlay-editor-dir", type=Path, default=Path("artifacts/overlay_georef_editor"))
    args = parser.parse_args()

    georef_json = json.loads(args.georef_json.read_text(encoding="utf-8"))
    page_no = int(georef_json["page_no"])
    frame_id = str(georef_json["frame_id"])

    frame_row = frame_row_by_key(args.frames, page_no, frame_id)
    pdf_bbox = frame_bbox_from_json_or_csv(georef_json, frame_row)
    params = build_projective_params(georef_json, pdf_bbox)

    features = load_routes(args.routes, page_no, frame_id)
    transformed_features: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    created_at = datetime.now(timezone.utc).isoformat()
    for feature in features:
        route_id = str(feature.get("properties", {}).get("route_id", "unknown_route"))
        geometry, issues = transform_geometry(feature["geometry"], params)
        if geometry is None:
            review_rows.append({"route_id": route_id, "status": "failed", "reason": "|".join(issues)})
            continue
        status = "review" if issues else "ok"
        props = dict(feature.get("properties", {}))
        props["coordinate_space"] = "wgs84_lonlat"
        props["image_georef_transform"] = "projective"
        props["image_georef_created_at"] = created_at
        props["needs_manual_review"] = bool(props.get("needs_manual_review")) or bool(issues)
        props["review_reasons"] = "|".join(sorted(set(filter(None, [props.get("review_reasons", ""), *issues])))).strip("|")
        transformed_features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": props,
            }
        )
        review_rows.append({"route_id": route_id, "status": status, "reason": "|".join(issues)})

    ensure_dir(args.out_dir)
    debug_dir = args.out_dir / "debug"
    ensure_dir(debug_dir)

    transformed_geojson = {"type": "FeatureCollection", "features": transformed_features}
    write_geojson(args.out_dir / "transformed_routes.geojson", transformed_features)
    write_csv(args.out_dir / "review_status.csv", review_rows)

    map_image = args.overlay_editor_dir / "images" / f"{frame_id}_map.png"
    redlines_image = args.overlay_editor_dir / "images" / f"{frame_id}_redlines.png"
    map_image_path = None
    redlines_image_path = None
    if map_image.exists():
        map_image_path = Path("../../../overlay_georef_editor/images") / map_image.name
        map_image_path = map_image_path.as_posix()
    if redlines_image.exists():
        redlines_image_path = Path("../../../overlay_georef_editor/images") / redlines_image.name
        redlines_image_path = redlines_image_path.as_posix()

    debug_html = build_debug_html(
        token=read_mapbox_token(args.env_file),
        transformed_geojson=transformed_geojson,
        georef_json=georef_json,
        map_image_path=map_image_path,
        redlines_image_path=redlines_image_path,
    )
    (debug_dir / "index.html").write_text(debug_html, encoding="utf-8")

    print(f"Wrote {len(transformed_features)} transformed routes to {args.out_dir}")


if __name__ == "__main__":
    main()
