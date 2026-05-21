#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MAPBOX_SCALE = 0.01
EARTH_RADIUS_M = 6378137.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_mapbox_token() -> str:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MAPBOX_ACCESS_TOKEN="):
                return line.partition("=")[2].strip()
    return ""


def synthetic_to_pdf(coord: list[float]) -> tuple[float, float]:
    return (coord[0] / MAPBOX_SCALE, -coord[1] / MAPBOX_SCALE)


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
    scale = np.trace(np.diag(d) @ s) / variance
    translation = tgt_mean - (scale * (rotation @ src_mean))
    return {"rotation": rotation, "scale": scale, "translation": translation}


def apply_similarity(points: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    rotation = params["rotation"]
    scale = float(params["scale"])
    translation = params["translation"]
    return ((scale * (rotation @ points.T)).T) + translation


def load_manual_gcps(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def frame_anchor_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in data.get("gcps", []) if row.get("role") == "frame_anchor"]


def load_routes(path: Path, page_no: int, frame_id: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        feature for feature in data["features"]
        if int(feature["properties"]["page_no"]) == page_no
        and feature["properties"].get("frame_id") == frame_id
    ]


def load_step5_routes(path: Path, page_no: int, frame_id: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        feature for feature in data["features"]
        if int(feature["properties"]["page_no"]) == page_no
        and feature["properties"].get("frame_id") == frame_id
    ]


def transform_route_geometry(
    geometry: dict[str, Any],
    params: dict[str, Any],
    page_height: float,
    ref_lon: float,
    ref_lat: float,
) -> dict[str, Any]:
    def transform_line(line: list[list[float]]) -> list[list[float]]:
        pdf_points = np.array([synthetic_to_pdf(coord) for coord in line], dtype=float)
        source = np.column_stack([pdf_points[:, 0], page_height - pdf_points[:, 1]])
        xy = apply_similarity(source, params)
        lonlat = [local_equirect_to_lonlat(x, y, ref_lon, ref_lat) for x, y in xy]
        return [[round(float(lon), 7), round(float(lat), 7)] for lon, lat in lonlat]

    if geometry["type"] == "LineString":
        return {"type": "LineString", "coordinates": transform_line(geometry["coordinates"])}
    if geometry["type"] == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [transform_line(line) for line in geometry["coordinates"]]}
    raise ValueError(f"unsupported geometry type: {geometry['type']}")


def fit_manual_model(data: dict[str, Any]) -> tuple[dict[str, Any], float, float, list[dict[str, Any]]]:
    anchors = frame_anchor_rows(data)
    if len(anchors) < 2:
        raise SystemExit("frame_anchor が不足しています")
    page_height = float(data["page"]["rect_pt"]["y1"]) - float(data["page"]["rect_pt"]["y0"])
    ref_lon = sum(float(row["longitude"]) for row in anchors) / len(anchors)
    ref_lat = sum(float(row["latitude"]) for row in anchors) / len(anchors)
    source = np.array(
        [(float(row["pdf_x"]), page_height - float(row["raw_pdf_y_top_left"])) for row in anchors],
        dtype=float,
    )
    target = np.array(
        [lonlat_to_local_equirect(float(row["longitude"]), float(row["latitude"]), ref_lon, ref_lat) for row in anchors],
        dtype=float,
    )
    params = fit_similarity(source, target)
    fitted = apply_similarity(source, params)
    fit_errors = [float(np.linalg.norm(fitted[i] - target[i])) for i in range(len(anchors))]

    residual_rows: list[dict[str, Any]] = []
    loocv_errors: list[float] = []
    for index, row in enumerate(anchors):
        train_idx = [i for i in range(len(anchors)) if i != index]
        train_source = source[train_idx]
        train_target = target[train_idx]
        train_params = fit_similarity(train_source, train_target)
        loocv_pred = apply_similarity(source[index:index + 1], train_params)[0]
        loocv_error = float(np.linalg.norm(loocv_pred - target[index]))
        loocv_errors.append(loocv_error)
        pred_lon, pred_lat = local_equirect_to_lonlat(fitted[index][0], fitted[index][1], ref_lon, ref_lat)
        residual_rows.append(
            {
                "index": row["index"],
                "name": row["name"],
                "fit_residual_m": round(fit_errors[index], 3),
                "loocv_residual_m": round(loocv_error, 3),
                "target_longitude": round(float(row["longitude"]), 7),
                "target_latitude": round(float(row["latitude"]), 7),
                "fit_pred_longitude": round(pred_lon, 7),
                "fit_pred_latitude": round(pred_lat, 7),
            }
        )

    fit_rmse = math.sqrt(sum(err * err for err in fit_errors) / len(fit_errors))
    loocv_rmse = math.sqrt(sum(err * err for err in loocv_errors) / len(loocv_errors))
    return params, ref_lon, ref_lat, [
        {
            "fit_rmse_m": round(fit_rmse, 3),
            "fit_max_m": round(max(fit_errors), 3),
            "loocv_rmse_m": round(loocv_rmse, 3),
            "loocv_max_m": round(max(loocv_errors), 3),
            "gcp_count": len(anchors),
        },
        *residual_rows,
    ]


def build_gcp_points_geojson(data: dict[str, Any], params: dict[str, Any], ref_lon: float, ref_lat: float) -> dict[str, Any]:
    anchors = frame_anchor_rows(data)
    page_height = float(data["page"]["rect_pt"]["y1"]) - float(data["page"]["rect_pt"]["y0"])
    source = np.array(
        [(float(row["pdf_x"]), page_height - float(row["raw_pdf_y_top_left"])) for row in anchors],
        dtype=float,
    )
    fitted = apply_similarity(source, params)
    features: list[dict[str, Any]] = []
    for idx, row in enumerate(anchors):
        pred_lon, pred_lat = local_equirect_to_lonlat(fitted[idx][0], fitted[idx][1], ref_lon, ref_lat)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]},
                "properties": {
                    "index": row["index"],
                    "name": row["name"],
                    "point_kind": "target",
                },
            }
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pred_lon, pred_lat]},
                "properties": {
                    "index": row["index"],
                    "name": row["name"],
                    "point_kind": "fit_pred",
                },
            }
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(row["longitude"]), float(row["latitude"])], [pred_lon, pred_lat]],
                },
                "properties": {
                    "index": row["index"],
                    "name": row["name"],
                    "point_kind": "residual_vector",
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: Path, features: list[dict[str, Any]] | dict[str, Any]) -> None:
    if isinstance(features, dict):
        data = features
    else:
        data = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_debug_html(token: str, summary: dict[str, Any]) -> str:
    config = json.dumps(
        {
            "mapboxToken": token,
            "gcpCount": summary["gcp_count"],
            "fitRmseM": summary["fit_rmse_m"],
            "loocvRmseM": summary["loocv_rmse_m"],
            "cacheBuster": f"gcp{summary['gcp_count']}_fit{summary['fit_rmse_m']}_loocv{summary['loocv_rmse_m']}",
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual Frame Override Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    html, body, #map {{ margin: 0; height: 100%; }}
    #panel {{
      position: absolute;
      top: 12px; left: 12px; z-index: 10;
      width: 320px; max-height: calc(100% - 24px); overflow: auto;
      padding: 12px 14px; border-radius: 12px;
      background: rgba(255,250,241,0.94); border: 1px solid #d9cfbb;
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
      color: #201d19; font-size: 12px; line-height: 1.5;
    }}
    .metric {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #eadfcb; }}
    .metric strong {{ display: block; font-size: 18px; }}
    button {{
      width: 100%; margin-bottom: 8px; padding: 8px 10px; border-radius: 8px;
      border: 1px solid #d9cfbb; background: white; cursor: pointer;
    }}
  </style>
</head>
<body>
  <div id="panel">
    <div class="metric"><strong>061_f06 Override</strong><span id="modelLabel"></span></div>
    <button id="showManual">Manual Override</button>
    <button id="showStep5">Original Step 5</button>
    <button id="showBoth">Both</button>
    <div id="summary"></div>
  </div>
  <div id="map"></div>
  <script>window.DEBUG_CONFIG = {config};</script>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js"></script>
  <script>
    const debugConfig = window.DEBUG_CONFIG || {{}};
    mapboxgl.accessToken = debugConfig.mapboxToken || '';
    const map = new mapboxgl.Map({{
      container: 'map',
      style: 'mapbox://styles/mapbox/outdoors-v12',
      center: [133.783, 34.224],
      zoom: 13
    }});
    const state = {{ manual: null, step5: null, gcps: null }};
    document.getElementById('modelLabel').textContent =
      `manual ${{debugConfig.gcpCount}}-point / y_flipped / similarity / LOOCV ${{debugConfig.loocvRmseM}}m`;
    async function load() {{
      const suffix = `?v=${{encodeURIComponent(debugConfig.cacheBuster || String(Date.now()))}}`;
      const [manualRes, step5Res, gcpsRes] = await Promise.all([
        fetch('manual_override_routes.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('original_step5_routes.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('gcp_points.geojson' + suffix, {{ cache: 'no-store' }})
      ]);
      state.manual = await manualRes.json();
      state.step5 = await step5Res.json();
      state.gcps = await gcpsRes.json();
      render('manual');
    }}
    function ensureSources() {{
      if (map.getSource('manual')) return;
      map.addSource('manual', {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      map.addSource('step5', {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      map.addSource('gcps', {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      map.addLayer({{
        id: 'step5-line', type: 'line', source: 'step5',
        paint: {{ 'line-color': '#bf5a19', 'line-width': 4, 'line-dasharray': [1.4, 1.1], 'line-opacity': 0.7 }}
      }});
      map.addLayer({{
        id: 'manual-line', type: 'line', source: 'manual',
        paint: {{ 'line-color': '#cf1b1b', 'line-width': 4.5, 'line-opacity': 0.95 }}
      }});
      map.addLayer({{
        id: 'gcp-vectors', type: 'line', source: 'gcps',
        filter: ['==', ['get', 'point_kind'], 'residual_vector'],
        paint: {{ 'line-color': '#146b78', 'line-width': 2 }}
      }});
      map.addLayer({{
        id: 'gcp-points', type: 'circle', source: 'gcps',
        filter: ['!=', ['get', 'point_kind'], 'residual_vector'],
        paint: {{
          'circle-radius': 6,
          'circle-color': ['case', ['==', ['get', 'point_kind'], 'target'], '#1f7a1f', '#146b78'],
          'circle-stroke-color': '#fff',
          'circle-stroke-width': 1.5
        }}
      }});
      map.addLayer({{
        id: 'gcp-labels', type: 'symbol', source: 'gcps',
        filter: ['==', ['get', 'point_kind'], 'target'],
        layout: {{ 'text-field': ['to-string', ['get', 'index']], 'text-offset': [0, 1.1], 'text-size': 11 }},
        paint: {{ 'text-color': '#111', 'text-halo-color': '#fff', 'text-halo-width': 1.2 }}
      }});
    }}
    function render(mode) {{
      if (!map.isStyleLoaded()) return;
      ensureSources();
      map.getSource('gcps').setData(state.gcps);
      map.getSource('manual').setData(mode === 'step5' ? {{ type:'FeatureCollection', features:[] }} : state.manual);
      map.getSource('step5').setData(mode === 'manual' ? {{ type:'FeatureCollection', features:[] }} : state.step5);
      const features = [];
      if (mode !== 'step5') features.push(...state.manual.features);
      if (mode !== 'manual') features.push(...state.step5.features);
      const bounds = new mapboxgl.LngLatBounds();
      for (const feat of features) {{
        const lines = feat.geometry.type === 'LineString' ? [feat.geometry.coordinates] : feat.geometry.coordinates;
        for (const line of lines) {{
          for (const coord of line) bounds.extend(coord);
        }}
      }}
      for (const feat of state.gcps.features) {{
        if (feat.geometry.type === 'Point') bounds.extend(feat.geometry.coordinates);
      }}
      if (!bounds.isEmpty()) map.fitBounds(bounds, {{ padding: 60, maxZoom: 15 }});
      document.getElementById('summary').textContent =
        mode === 'manual' ? 'manual override only' : mode === 'step5' ? 'original step5 only' : 'manual override + original step5';
    }}
    map.on('load', () => {{
      document.getElementById('showManual').onclick = () => render('manual');
      document.getElementById('showStep5').onclick = () => render('step5');
      document.getElementById('showBoth').onclick = () => render('both');
      load();
    }});
  </script>
</body>
</html>
"""


def write_report(path: Path, summary: dict[str, Any], route_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Manual Frame Override",
        "",
        "- Frame: `061_f06`",
        "- Model: `y_flipped + similarity`",
        f"- GCP count: `{summary['gcp_count']}`",
        f"- Fit RMSE: `{summary['fit_rmse_m']}` m",
        f"- LOOCV RMSE: `{summary['loocv_rmse_m']}` m",
        "",
        "## Route Count",
        f"- Manual override routes: `{len(route_rows)}`",
        "",
        "## Outputs",
        "- `manual_override_routes.geojson`",
        "- `original_step5_routes.geojson`",
        "- `gcp_points.geojson`",
        "- `route_summary.csv`",
        "- `debug/index.html`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual y_flipped similarity override to one frame")
    parser.add_argument("--manual-json", type=Path, default=Path("data/manual_gcps/page_061_061_f06.json"))
    parser.add_argument("--step3-routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--step5-routes", type=Path, default=Path("artifacts/step5/transformed_routes.geojson"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/frame_override"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    data = load_manual_gcps(args.manual_json)
    page_no = int(data["page_no"])
    frame_id = data["frame_id"]
    params, ref_lon, ref_lat, diagnostics = fit_manual_model(data)
    summary = diagnostics[0]
    residual_rows = diagnostics[1:]

    step3_routes = load_routes(args.step3_routes, page_no, frame_id)
    manual_features: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for feature in step3_routes:
        geometry = transform_route_geometry(
            geometry=feature["geometry"],
            params=params,
            page_height=float(data["page"]["rect_pt"]["y1"]) - float(data["page"]["rect_pt"]["y0"]),
            ref_lon=ref_lon,
            ref_lat=ref_lat,
        )
        props = dict(feature["properties"])
        props.update(
            {
                "coordinate_space": "wgs84_manual_override",
                "transform_model": "manual_y_flipped_similarity",
                "gcp_count": summary["gcp_count"],
                "fit_rmse_m": summary["fit_rmse_m"],
                "loocv_rmse_m": summary["loocv_rmse_m"],
            }
        )
        manual_features.append({"type": "Feature", "geometry": geometry, "properties": props})
        route_rows.append(
            {
                "route_id": props["route_id"],
                "style_class": props.get("style_class", ""),
                "confidence": props.get("confidence", ""),
                "needs_manual_review": props.get("needs_manual_review", ""),
            }
        )

    step5_features = load_step5_routes(args.step5_routes, page_no, frame_id)
    write_geojson(args.out_dir / "manual_override_routes.geojson", manual_features)
    write_geojson(args.out_dir / "original_step5_routes.geojson", step5_features)
    write_geojson(args.out_dir / "gcp_points.geojson", build_gcp_points_geojson(data, params, ref_lon, ref_lat))
    write_csv(args.out_dir / "residual_summary.csv", residual_rows)
    write_csv(args.out_dir / "route_summary.csv", route_rows)
    write_report(args.out_dir / "report.md", summary, route_rows)

    debug_dir = args.out_dir / "debug"
    ensure_dir(debug_dir)
    for name in ["manual_override_routes.geojson", "original_step5_routes.geojson", "gcp_points.geojson"]:
        target = debug_dir / name
        source = args.out_dir / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "index.html").write_text(build_debug_html(read_mapbox_token(), summary), encoding="utf-8")


if __name__ == "__main__":
    main()
