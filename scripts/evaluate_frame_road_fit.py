#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


EARTH_RADIUS_M = 6378137.0
SAMPLE_INTERVAL_M = 10.0
FAILED_THRESHOLD_M = 50.0
SNAP_THRESHOLD_M = 35.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SNAP_ALLOWED_HIGHWAYS = {
    "motorway_link",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "pedestrian",
    "footway",
    "path",
    "track",
    "cycleway",
    "steps",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_mapbox_token() -> str:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MAPBOX_ACCESS_TOKEN="):
                return line.partition("=")[2].strip()
    return ""


def lonlat_to_local_equirect(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return (x, y)


def local_equirect_to_lonlat(x: float, y: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    return (lon, lat)


def bbox_from_features(features: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        geometry = feature["geometry"]
        lines = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        for line in lines:
            for lon, lat in line:
                xs.append(lon)
                ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys))


def load_geojson_features(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["features"]


def load_geojson(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_overpass_query(bbox: tuple[float, float, float, float], pad_deg: float = 0.003) -> str:
    west, south, east, north = bbox[0] - pad_deg, bbox[1] - pad_deg, bbox[2] + pad_deg, bbox[3] + pad_deg
    return f"""
[out:json][timeout:60];
(
  way["highway"]({south},{west},{north},{east});
);
out geom;
"""


def fetch_overpass_json(query: str) -> dict[str, Any]:
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(OVERPASS_URL, data=payload, headers={"User-Agent": "takram-road-fit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def overpass_to_geojson(data: dict[str, Any]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
        coords = [[node["lon"], node["lat"]] for node in element.get("geometry", [])]
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "osm_way_id": element.get("id"),
                    "highway": element.get("tags", {}).get("highway", ""),
                    "name": element.get("tags", {}).get("name", ""),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def segment_projection(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    seg_len2 = (dx * dx) + (dy * dy)
    if seg_len2 <= 1e-12:
        dist = math.dist(point, start)
        return start, dist, 0.0
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj = (start[0] + t * dx, start[1] + t * dy)
    dist = math.dist(point, proj)
    return proj, dist, t


def densify_line(
    coords_lonlat: list[list[float]],
    ref_lon: float,
    ref_lat: float,
    sample_interval_m: float = SAMPLE_INTERVAL_M,
) -> list[dict[str, Any]]:
    local = [lonlat_to_local_equirect(lon, lat, ref_lon, ref_lat) for lon, lat in coords_lonlat]
    samples: list[dict[str, Any]] = []
    cumulative = 0.0
    if not local:
        return samples
    samples.append({"xy": local[0], "lonlat": coords_lonlat[0], "along_m": 0.0})
    for index in range(1, len(local)):
        start = local[index - 1]
        end = local[index]
        seg_len = math.dist(start, end)
        if seg_len <= 1e-9:
            continue
        next_target = sample_interval_m
        while next_target < seg_len:
            t = next_target / seg_len
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            lon, lat = local_equirect_to_lonlat(x, y, ref_lon, ref_lat)
            samples.append({"xy": (x, y), "lonlat": [lon, lat], "along_m": cumulative + next_target})
            next_target += sample_interval_m
        cumulative += seg_len
        samples.append({"xy": end, "lonlat": coords_lonlat[index], "along_m": cumulative})
    return samples


def prepare_road_segments(roads: dict[str, Any], ref_lon: float, ref_lat: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for feature in roads["features"]:
        coords = feature["geometry"]["coordinates"]
        local = [lonlat_to_local_equirect(lon, lat, ref_lon, ref_lat) for lon, lat in coords]
        for index in range(1, len(local)):
            segments.append(
                {
                    "start": local[index - 1],
                    "end": local[index],
                    "highway": feature["properties"].get("highway", ""),
                    "name": feature["properties"].get("name", ""),
                    "osm_way_id": feature["properties"].get("osm_way_id", ""),
                }
            )
    return segments


def nearest_road(point_xy: tuple[float, float], road_segments: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for segment in road_segments:
        proj, dist, t = segment_projection(point_xy, segment["start"], segment["end"])
        if best is None or dist < best["distance_m"]:
            best = {
                "proj_xy": proj,
                "distance_m": dist,
                "highway": segment["highway"],
                "name": segment["name"],
                "osm_way_id": segment["osm_way_id"],
                "t": t,
            }
    return best or {
        "proj_xy": point_xy,
        "distance_m": float("inf"),
        "highway": "",
        "name": "",
        "osm_way_id": "",
        "t": 0.0,
    }


def sample_routes_against_roads(
    routes: list[dict[str, Any]],
    road_segments: list[dict[str, Any]],
    ref_lon: float,
    ref_lat: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    failed_features: list[dict[str, Any]] = []
    snapped_features: list[dict[str, Any]] = []
    route_summary_rows: list[dict[str, Any]] = []

    for feature in routes:
        route_id = feature["properties"]["route_id"]
        geometry = feature["geometry"]
        lines = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        snapped_lines: list[list[list[float]]] = []
        failed_run: list[list[float]] = []
        route_distances: list[float] = []
        snapped_count = 0
        total_count = 0

        for line_index, line in enumerate(lines):
            samples = densify_line(line, ref_lon, ref_lat)
            snapped_line: list[list[float]] = []
            for sample_index, sample in enumerate(samples):
                nearest = nearest_road(sample["xy"], road_segments)
                distance_m = float(nearest["distance_m"])
                route_distances.append(distance_m)
                total_count += 1
                should_snap = (
                    distance_m <= SNAP_THRESHOLD_M
                    and nearest["highway"] in SNAP_ALLOWED_HIGHWAYS
                )
                if should_snap:
                    snapped_count += 1
                    lon, lat = local_equirect_to_lonlat(nearest["proj_xy"][0], nearest["proj_xy"][1], ref_lon, ref_lat)
                    snapped_line.append([round(lon, 7), round(lat, 7)])
                else:
                    snapped_line.append([round(sample["lonlat"][0], 7), round(sample["lonlat"][1], 7)])

                sample_rows.append(
                    {
                        "route_id": route_id,
                        "line_index": line_index,
                        "sample_index": sample_index,
                        "longitude": round(sample["lonlat"][0], 7),
                        "latitude": round(sample["lonlat"][1], 7),
                        "along_m": round(sample["along_m"], 3),
                        "nearest_road_distance_m": round(distance_m, 3),
                        "nearest_highway": nearest["highway"],
                        "nearest_name": nearest["name"],
                        "osm_way_id": nearest["osm_way_id"],
                        "snap_applied": should_snap,
                        "failed_threshold": distance_m > FAILED_THRESHOLD_M,
                    }
                )

                original_point = [round(sample["lonlat"][0], 7), round(sample["lonlat"][1], 7)]
                if distance_m > FAILED_THRESHOLD_M:
                    failed_run.append(original_point)
                else:
                    if len(failed_run) >= 2:
                        failed_features.append(
                            {
                                "type": "Feature",
                                "geometry": {"type": "LineString", "coordinates": failed_run[:]},
                                "properties": {
                                    "route_id": route_id,
                                    "line_index": line_index,
                                    "status": "failed",
                                    "needs_review": True,
                                },
                            }
                        )
                    failed_run = []

            if len(failed_run) >= 2:
                failed_features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": failed_run[:]},
                        "properties": {
                            "route_id": route_id,
                            "line_index": line_index,
                            "status": "failed",
                            "needs_review": True,
                        },
                    }
                )
            snapped_lines.append(snapped_line)

        snapped_geometry = {"type": "LineString", "coordinates": snapped_lines[0]} if len(snapped_lines) == 1 else {"type": "MultiLineString", "coordinates": snapped_lines}
        max_distance = max(route_distances) if route_distances else 0.0
        mean_distance = sum(route_distances) / len(route_distances) if route_distances else 0.0
        snap_ratio = snapped_count / total_count if total_count else 0.0
        needs_review = max_distance > FAILED_THRESHOLD_M
        props = dict(feature["properties"])
        props.update(
            {
                "road_eval_mean_distance_m": round(mean_distance, 3),
                "road_eval_max_distance_m": round(max_distance, 3),
                "road_eval_snap_ratio": round(snap_ratio, 4),
                "road_eval_failed": needs_review,
                "road_eval_needs_review": needs_review,
                "road_eval_status": "needs_review" if needs_review else "ok",
                "road_eval_model": "osm_nearest_road",
                "georef_model": "y_flipped_similarity",
            }
        )
        snapped_features.append({"type": "Feature", "geometry": snapped_geometry, "properties": props})
        route_summary_rows.append(
            {
                "route_id": route_id,
                "style_class": props.get("style_class", ""),
                "mean_distance_m": round(mean_distance, 3),
                "max_distance_m": round(max_distance, 3),
                "snap_ratio": round(snap_ratio, 4),
                "failed": needs_review,
                "status": "needs_review" if needs_review else "ok",
            }
        )

    return sample_rows, failed_features, snapped_features, route_summary_rows


def write_geojson(path: Path, features: list[dict[str, Any]] | dict[str, Any]) -> None:
    data = features if isinstance(features, dict) else {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_debug_html(token: str, route_summary_rows: list[dict[str, Any]]) -> str:
    failed_count = sum(1 for row in route_summary_rows if row["failed"])
    config = json.dumps(
        {
            "mapboxToken": token,
            "failedCount": failed_count,
            "routeCount": len(route_summary_rows),
            "cacheBuster": f"routes{len(route_summary_rows)}_failed{failed_count}",
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Road Fit Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    html, body, #map {{ margin: 0; height: 100%; }}
    #panel {{
      position: absolute; top: 12px; left: 12px; z-index: 10;
      width: 360px; max-height: calc(100% - 24px); overflow: auto;
      padding: 12px 14px; border-radius: 12px;
      background: rgba(255,250,241,0.94); border: 1px solid #d9cfbb;
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
      color: #201d19; font-size: 12px; line-height: 1.5;
    }}
    .metric {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #eadfcb; }}
    .metric strong {{ display: block; font-size: 18px; }}
    button {{ width: 100%; margin-bottom: 8px; padding: 8px 10px; border-radius: 8px; border: 1px solid #d9cfbb; background: white; cursor: pointer; }}
    .list {{ margin-top: 10px; }}
    .item {{ background: white; border: 1px solid #eadfcb; border-radius: 10px; padding: 8px 10px; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <div id="panel">
    <div class="metric">
      <strong>061_f06 Road Fit</strong>
      manual georef route count <span id="routeCount"></span> / failed routes <span id="failedCount"></span>
    </div>
    <button id="showManual">Manual Red Line</button>
    <button id="showSnapped">Snapped Trial</button>
    <button id="showBoth">Both</button>
    <button id="showFailed">Failed Segments</button>
    <div id="summary" class="list"></div>
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
      center: [133.79, 34.222],
      zoom: 13
    }});
    const state = {{ manual: null, snapped: null, failed: null, roads: null, gcps: null, routes: [] }};
    document.getElementById('routeCount').textContent = String(debugConfig.routeCount || 0);
    document.getElementById('failedCount').textContent = String(debugConfig.failedCount || 0);
    async function load() {{
      const suffix = `?v=${{encodeURIComponent(debugConfig.cacheBuster || String(Date.now()))}}`;
      const [manualRes, snappedRes, failedRes, roadsRes, gcpsRes, summaryRes] = await Promise.all([
        fetch('manual_override_routes.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('snapped_routes.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('failed_segments.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('osm_roads.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('gcp_points.geojson' + suffix, {{ cache: 'no-store' }}),
        fetch('route_summary.json' + suffix, {{ cache: 'no-store' }})
      ]);
      state.manual = await manualRes.json();
      state.snapped = await snappedRes.json();
      state.failed = await failedRes.json();
      state.roads = await roadsRes.json();
      state.gcps = await gcpsRes.json();
      state.routes = await summaryRes.json();
      render('both');
    }}
    function ensureSources() {{
      if (map.getSource('manual')) return;
      for (const id of ['manual','snapped','failed','roads','gcps']) {{
        map.addSource(id, {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      }}
      map.addLayer({{ id:'roads-line', type:'line', source:'roads', paint:{{ 'line-color':'#69737d', 'line-width':2, 'line-opacity':0.5 }} }});
      map.addLayer({{ id:'manual-line', type:'line', source:'manual', paint:{{ 'line-color':'#cf1b1b', 'line-width':4.5 }} }});
      map.addLayer({{ id:'snapped-line', type:'line', source:'snapped', paint:{{ 'line-color':'#0e7a7a', 'line-width':3.5, 'line-opacity':0.9 }} }});
      map.addLayer({{ id:'failed-line', type:'line', source:'failed', paint:{{ 'line-color':'#d07a00', 'line-width':6, 'line-dasharray':[1.2,0.8] }} }});
      map.addLayer({{ id:'gcp-vectors', type:'line', source:'gcps', filter:['==',['get','point_kind'],'residual_vector'], paint:{{ 'line-color':'#146b78', 'line-width':2 }} }});
      map.addLayer({{ id:'gcp-points', type:'circle', source:'gcps', filter:['!=',['get','point_kind'],'residual_vector'], paint:{{ 'circle-radius':6, 'circle-color':['case',['==',['get','point_kind'],'target'],'#1f7a1f','#146b78'], 'circle-stroke-color':'#fff', 'circle-stroke-width':1.5 }} }});
      map.addLayer({{ id:'gcp-labels', type:'symbol', source:'gcps', filter:['==',['get','point_kind'],'target'], layout:{{ 'text-field':['to-string',['get','index']], 'text-offset':[0,1.1], 'text-size':11 }}, paint:{{ 'text-color':'#111','text-halo-color':'#fff','text-halo-width':1.2 }} }});
    }}
    function setData(mode) {{
      map.getSource('roads').setData(state.roads);
      map.getSource('gcps').setData(state.gcps);
      map.getSource('manual').setData(mode === 'snapped' ? {{ type:'FeatureCollection', features:[] }} : state.manual);
      map.getSource('snapped').setData(mode === 'manual' ? {{ type:'FeatureCollection', features:[] }} : state.snapped);
      map.getSource('failed').setData(mode === 'failed' || mode === 'both' ? state.failed : {{ type:'FeatureCollection', features:[] }});
    }}
    function renderList() {{
      document.getElementById('summary').innerHTML = state.routes.map(row => `
        <div class="item">
          <strong>${{row.route_id}}</strong><br>
          mean=${{row.mean_distance_m}}m / max=${{row.max_distance_m}}m / snap=${{row.snap_ratio}} / failed=${{row.failed}}
        </div>
      `).join('');
    }}
    function fitBounds() {{
      const bounds = new mapboxgl.LngLatBounds();
      for (const collection of [state.manual, state.snapped, state.failed, state.gcps]) {{
        for (const feat of collection.features) {{
          if (feat.geometry.type === 'Point') bounds.extend(feat.geometry.coordinates);
          else if (feat.geometry.type === 'LineString') feat.geometry.coordinates.forEach(coord => bounds.extend(coord));
          else feat.geometry.coordinates.forEach(line => line.forEach(coord => bounds.extend(coord)));
        }}
      }}
      if (!bounds.isEmpty()) map.fitBounds(bounds, {{ padding: 60, maxZoom: 15 }});
    }}
    function render(mode) {{
      if (!map.isStyleLoaded()) return;
      ensureSources();
      setData(mode);
      renderList();
      fitBounds();
    }}
    map.on('load', () => {{
      document.getElementById('showManual').onclick = () => render('manual');
      document.getElementById('showSnapped').onclick = () => render('snapped');
      document.getElementById('showBoth').onclick = () => render('both');
      document.getElementById('showFailed').onclick = () => render('failed');
      load();
    }});
  </script>
</body>
</html>
"""


def write_report(path: Path, route_summary_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    failed_routes = [row for row in route_summary_rows if row["failed"]]
    lines = [
        "# Road Fit Report",
        "",
        "- Frame: `061_f06`",
        "- georef_model: `y_flipped_similarity`",
        f"- Route count: `{len(route_summary_rows)}`",
        f"- Failed route count (>50m sampled distance): `{len(failed_routes)}`",
        f"- Sample count: `{len(sample_rows)}`",
        "",
        "## Outputs",
        "- `osm_roads.geojson`",
        "- `route_samples.csv`",
        "- `failed_segments.geojson`",
        "- `snapped_routes.geojson`",
        "- `route_summary.csv`",
        "- `debug/index.html`",
        "",
        "## Notes",
        "- 道路距離評価は OSM `highway=*` を nearest road として計算しています。",
        "- snap は nearest road が `SNAP_ALLOWED_HIGHWAYS` かつ 35m 以内の時だけ適用しています。",
        "- 50m 超サンプルを含む区間は `failed` / `needs_review` 候補です。",
        "- OSM に存在しない山道・参道・徒歩道は無理に吸着していません。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate manual-georeferenced routes against OSM road network")
    parser.add_argument("--manual-routes", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/frame_override/manual_override_routes.geojson"))
    parser.add_argument("--gcp-points", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/frame_override/gcp_points.geojson"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval"))
    parser.add_argument("--fetch-osm", action="store_true")
    parser.add_argument("--osm-json-cache", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval/osm_overpass_raw.json"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    manual_geojson = load_geojson(args.manual_routes)
    features = manual_geojson["features"]
    bbox = bbox_from_features(features)
    ref_lon = (bbox[0] + bbox[2]) / 2.0
    ref_lat = (bbox[1] + bbox[3]) / 2.0

    if args.fetch_osm or not args.osm_json_cache.exists():
        overpass = fetch_overpass_json(build_overpass_query(bbox))
        ensure_dir(args.osm_json_cache.parent)
        args.osm_json_cache.write_text(json.dumps(overpass, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        overpass = json.loads(args.osm_json_cache.read_text(encoding="utf-8"))

    roads_geojson = overpass_to_geojson(overpass)
    road_segments = prepare_road_segments(roads_geojson, ref_lon, ref_lat)
    sample_rows, failed_features, snapped_features, route_summary_rows = sample_routes_against_roads(features, road_segments, ref_lon, ref_lat)

    write_geojson(args.out_dir / "osm_roads.geojson", roads_geojson)
    write_csv(args.out_dir / "route_samples.csv", sample_rows)
    write_geojson(args.out_dir / "failed_segments.geojson", failed_features)
    write_geojson(args.out_dir / "snapped_routes.geojson", snapped_features)
    write_csv(args.out_dir / "route_summary.csv", route_summary_rows)
    (args.out_dir / "route_summary.json").write_text(json.dumps(route_summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out_dir / "report.md", route_summary_rows, sample_rows)

    debug_dir = args.out_dir / "debug"
    ensure_dir(debug_dir)
    for source_name, source_path in [
        ("manual_override_routes.geojson", args.manual_routes),
        ("snapped_routes.geojson", args.out_dir / "snapped_routes.geojson"),
        ("failed_segments.geojson", args.out_dir / "failed_segments.geojson"),
        ("osm_roads.geojson", args.out_dir / "osm_roads.geojson"),
        ("gcp_points.geojson", args.gcp_points),
        ("route_summary.json", args.out_dir / "route_summary.json"),
    ]:
        (debug_dir / source_name).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "index.html").write_text(build_debug_html(read_mapbox_token(), route_summary_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
