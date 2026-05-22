#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

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


def load_geojson_by_route(path: Path) -> dict[str, dict[str, Any]]:
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    return {feature["properties"]["route_id"]: feature for feature in features}


def load_diagnosis(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lonlat_to_local_equirect(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return (x, y)


def local_equirect_to_lonlat(x: float, y: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    return (lon, lat)


def segment_projection(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    seg_len2 = (dx * dx) + (dy * dy)
    if seg_len2 <= 1e-12:
        return start, math.dist(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj = (start[0] + t * dx, start[1] + t * dy)
    return proj, math.dist(point, proj)


def geometry_lines(feature: dict[str, Any]) -> list[list[list[float]]]:
    geom = feature["geometry"]
    return [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]


def geometry_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for line in geometry_lines(feature):
        for lon, lat in line:
            xs.append(lon)
            ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys))


def load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_road_segments(roads_geojson: dict[str, Any], ref_lon: float, ref_lat: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for feature in roads_geojson["features"]:
        coords = feature["geometry"]["coordinates"]
        local = [lonlat_to_local_equirect(lon, lat, ref_lon, ref_lat) for lon, lat in coords]
        for index in range(1, len(local)):
            segments.append({"start": local[index - 1], "end": local[index]})
    return segments


def nearest_road_distance(point_lonlat: list[float], road_segments: list[dict[str, Any]], ref_lon: float, ref_lat: float) -> float:
    xy = lonlat_to_local_equirect(point_lonlat[0], point_lonlat[1], ref_lon, ref_lat)
    best = float("inf")
    for segment in road_segments:
        _, dist = segment_projection(xy, segment["start"], segment["end"])
        if dist < best:
            best = dist
    return best


def sample_linestring(coords: list[list[float]], interval_m: float = 10.0) -> list[list[float]]:
    if len(coords) < 2:
        return coords[:]
    ref_lon = sum(c[0] for c in coords) / len(coords)
    ref_lat = sum(c[1] for c in coords) / len(coords)
    local = [lonlat_to_local_equirect(lon, lat, ref_lon, ref_lat) for lon, lat in coords]
    out: list[list[float]] = [coords[0]]
    for idx in range(1, len(local)):
        start = local[idx - 1]
        end = local[idx]
        seg_len = math.dist(start, end)
        if seg_len <= 1e-9:
            continue
        next_target = interval_m
        while next_target < seg_len:
            t = next_target / seg_len
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            lon, lat = local_equirect_to_lonlat(x, y, ref_lon, ref_lat)
            out.append([lon, lat])
            next_target += interval_m
        out.append(coords[idx])
    return out


def feature_samples(feature: dict[str, Any], interval_m: float = 10.0) -> list[list[float]]:
    samples: list[list[float]] = []
    for line in geometry_lines(feature):
        line_samples = sample_linestring(line, interval_m=interval_m)
        if samples and line_samples:
            samples.extend(line_samples[1:])
        else:
            samples.extend(line_samples)
    return samples


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * q) - 1)))
    return float(ordered[idx])


def geometry_step_lengths(feature: dict[str, Any]) -> list[float]:
    lengths: list[float] = []
    for line in geometry_lines(feature):
        if len(line) < 2:
            continue
        ref_lon = sum(c[0] for c in line) / len(line)
        ref_lat = sum(c[1] for c in line) / len(line)
        local = [lonlat_to_local_equirect(lon, lat, ref_lon, ref_lat) for lon, lat in line]
        lengths.extend(math.dist(a, b) for a, b in zip(local, local[1:]))
    return lengths


def route_samples_by_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["route_id"], []).append(row)
    return rows


def append_reason(existing: str, reason: str) -> str:
    parts = [item for item in existing.split(",") if item]
    if reason and reason not in parts:
        parts.append(reason)
    return ",".join(parts)


def make_feature(geometry: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def build_debug_html(token: str) -> str:
    config = json.dumps({"mapboxToken": token}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Postprocess Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    html, body, #map {{ margin: 0; height: 100%; }}
    #panel {{
      position: absolute; top: 12px; left: 12px; z-index: 10;
      width: 360px; max-height: calc(100% - 24px); overflow: auto;
      padding: 12px 14px; border-radius: 12px;
      background: rgba(255,250,241,0.96); border: 1px solid #d9cfbb;
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
      color: #201d19; font-size: 12px; line-height: 1.5;
    }}
    .metric {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #eadfcb; }}
    .metric strong {{ display: block; font-size: 18px; }}
    button {{
      width: 100%; margin-bottom: 8px; padding: 8px 10px;
      border-radius: 8px; border: 1px solid #d9cfbb; background: white; cursor: pointer;
    }}
    .item {{
      background: white; border: 1px solid #eadfcb; border-radius: 10px;
      padding: 8px 10px; margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <div id="panel">
    <div class="metric">
      <strong>061_f06 Postprocess</strong>
      original / final / candidate comparison
    </div>
    <button id="showFinal">Final</button>
    <button id="showMatched">Matched Final</button>
    <button id="showOriginal">Original</button>
    <button id="showCandidates">Candidates</button>
    <button id="showAll">All</button>
    <div id="summary"></div>
  </div>
  <div id="map"></div>
  <script>window.DEBUG_CONFIG = {config};</script>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js"></script>
  <script>
    const token = window.DEBUG_CONFIG.mapboxToken || '';
    mapboxgl.accessToken = token;
    const map = new mapboxgl.Map({{
      container: 'map',
      style: 'mapbox://styles/mapbox/outdoors-v12',
      center: [133.79, 34.222],
      zoom: 13
    }});
    const state = {{ original: null, final: null, matched: null, candidates: null, osmMissing: null, manual: null, diagnosis: [], assessment: [] }};
    async function load() {{
      const [origRes, finalRes, matchedRes, candRes, osmRes, manualRes, diagRes, assessRes] = await Promise.all([
        fetch('original_routes.geojson', {{ cache: 'no-store' }}),
        fetch('final_routes.geojson', {{ cache: 'no-store' }}),
        fetch('final_routes_matched.geojson', {{ cache: 'no-store' }}),
        fetch('snap_candidates.geojson', {{ cache: 'no-store' }}),
        fetch('osm_missing_or_trail.geojson', {{ cache: 'no-store' }}),
        fetch('manual_edit_needed.geojson', {{ cache: 'no-store' }}),
        fetch('route_diagnosis.json', {{ cache: 'no-store' }}),
        fetch('candidate_assessment.json', {{ cache: 'no-store' }})
      ]);
      state.original = await origRes.json();
      state.final = await finalRes.json();
      state.matched = await matchedRes.json();
      state.candidates = await candRes.json();
      state.osmMissing = await osmRes.json();
      state.manual = await manualRes.json();
      state.diagnosis = await diagRes.json();
      state.assessment = await assessRes.json();
      render('all');
    }}
    function ensure() {{
      if (map.getSource('final')) return;
      for (const id of ['original','final','matched','candidates','osmMissing','manual']) {{
        map.addSource(id, {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      }}
      map.addLayer({{ id:'original-line', type:'line', source:'original', paint:{{ 'line-color':'#9aa1a8', 'line-width':3, 'line-opacity':0.6 }} }});
      map.addLayer({{ id:'final-line', type:'line', source:'final', paint:{{ 'line-color':'#cf1b1b', 'line-width':4 }} }});
      map.addLayer({{ id:'matched-line', type:'line', source:'matched', paint:{{ 'line-color':'#0b8f6d', 'line-width':4.5 }} }});
      map.addLayer({{ id:'candidate-line', type:'line', source:'candidates', paint:{{ 'line-color':'#0e7a7a', 'line-width':3.5, 'line-dasharray':[1.2,0.8] }} }});
      map.addLayer({{ id:'osm-line', type:'line', source:'osmMissing', paint:{{ 'line-color':'#c88b00', 'line-width':4 }} }});
      map.addLayer({{ id:'manual-line', type:'line', source:'manual', paint:{{ 'line-color':'#6b46c1', 'line-width':4.5 }} }});
    }}
    function setData(mode) {{
      map.getSource('original').setData(mode === 'original' || mode === 'all' ? state.original : {{ type:'FeatureCollection', features:[] }});
      map.getSource('final').setData(mode === 'final' || mode === 'all' ? state.final : {{ type:'FeatureCollection', features:[] }});
      map.getSource('matched').setData(mode === 'matched' || mode === 'all' ? state.matched : {{ type:'FeatureCollection', features:[] }});
      map.getSource('candidates').setData(mode === 'candidates' || mode === 'all' ? state.candidates : {{ type:'FeatureCollection', features:[] }});
      map.getSource('osmMissing').setData(mode === 'all' ? state.osmMissing : {{ type:'FeatureCollection', features:[] }});
      map.getSource('manual').setData(mode === 'all' ? state.manual : {{ type:'FeatureCollection', features:[] }});
    }}
    function renderSummary() {{
      const assessById = Object.fromEntries((state.assessment || []).map(row => [row.route_id, row]));
      document.getElementById('summary').innerHTML = state.diagnosis.map(row => {{
        const assess = assessById[row.route_id];
        const assessmentLines = assess ? [
          `candidate=${{assess.adopt_candidate ? 'adopt' : 'keep_original'}}`,
          `cand mean=${{assess.candidate_mean_road_dist_m}}m / p90=${{assess.candidate_p90_road_dist_m}}m / max=${{assess.candidate_max_road_dist_m}}m`,
          `disp p95=${{assess.snap_displacement_p95_m}}m / max=${{assess.snap_displacement_max_m}}m`,
          `reasons=${{assess.adoption_reasons || 'pass'}}`
        ].join('<br>') : '';
        return `
          <div class="item">
            <strong>${{row.route_id}}</strong><br>
            action=${{row.recommended_action}}<br>
            likely_issue=${{row.likely_issue}}<br>
            mean=${{row.mean_road_dist_m}}m / max=${{row.max_road_dist_m}}m
            ${{assessmentLines ? '<br>' + assessmentLines : ''}}
          </div>
        `;
      }}).join('');
    }}
    function fitBounds() {{
      const bounds = new mapboxgl.LngLatBounds();
      for (const collection of [state.final, state.original, state.matched, state.candidates]) {{
        if (!collection) continue;
        for (const feat of collection.features) {{
          const lines = feat.geometry.type === 'LineString' ? [feat.geometry.coordinates] : feat.geometry.coordinates;
          for (const line of lines) {{
            for (const coord of line) bounds.extend(coord);
          }}
        }}
      }}
      if (!bounds.isEmpty()) map.fitBounds(bounds, {{ padding: 60, maxZoom: 15 }});
    }}
    function render(mode) {{
      if (!map.isStyleLoaded()) return;
      ensure();
      setData(mode);
      renderSummary();
      fitBounds();
    }}
    map.on('load', () => {{
      document.getElementById('showFinal').onclick = () => render('final');
      document.getElementById('showMatched').onclick = () => render('matched');
      document.getElementById('showOriginal').onclick = () => render('original');
      document.getElementById('showCandidates').onclick = () => render('candidates');
      document.getElementById('showAll').onclick = () => render('all');
      load();
    }});
  </script>
</body>
</html>
"""


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, Any]], assessments: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["recommended_action"]] = counts.get(row["recommended_action"], 0) + 1
    adopted = [row for row in assessments if row["adopt_candidate"]]
    rejected = [row for row in assessments if not row["adopt_candidate"]]
    lines = [
        "# Postprocess Report",
        "",
        "- Frame: `061_f06`",
        f"- Route count: `{len(rows)}`",
        f"- Matched candidate adopted: `{len(adopted)}`",
        f"- Matched candidate rejected: `{len(rejected)}`",
        "",
        "## Action Counts",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: `{counts[key]}`")
    if assessments:
        lines.extend(["", "## Candidate Assessment"])
        for row in assessments:
            decision = "adopt" if row["adopt_candidate"] else "keep_original"
            lines.append(
                "- "
                f"`{row['route_id']}`: `{decision}` "
                f"(orig mean/p90/max={row['original_mean_road_dist_m']}/{row['original_p90_road_dist_m']}/{row['original_max_road_dist_m']}m, "
                f"cand mean/p90/max={row['candidate_mean_road_dist_m']}/{row['candidate_p90_road_dist_m']}/{row['candidate_max_road_dist_m']}m, "
                f"disp p95/max={row['snap_displacement_p95_m']}/{row['snap_displacement_max_m']}m, "
                f"reasons=`{row['adoption_reasons'] or 'pass'}`)"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- `final_routes.geojson` は元線保持です。",
            "- `final_routes_matched.geojson` は candidate が明確に良い route だけ置換しています。",
            "- `osm_missing_or_trail` は元線を保持し、snap していません。",
            "- `manual_edit_needed` は元線を保持し、needs_manual_review を維持しています。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_candidate(
    route_id: str,
    original_feature: dict[str, Any],
    candidate_feature: dict[str, Any],
    diagnosis_row: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    road_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    bbox = geometry_bbox(candidate_feature)
    ref_lon = (bbox[0] + bbox[2]) / 2.0
    ref_lat = (bbox[1] + bbox[3]) / 2.0

    candidate_points = feature_samples(candidate_feature, interval_m=10.0)
    candidate_dists = [nearest_road_distance(point, road_segments, ref_lon, ref_lat) for point in candidate_points]
    candidate_mean = sum(candidate_dists) / len(candidate_dists) if candidate_dists else 0.0
    candidate_p90 = percentile(candidate_dists, 0.9)
    candidate_max = max(candidate_dists) if candidate_dists else 0.0

    original_mean = float(diagnosis_row["mean_road_dist_m"])
    original_max = float(diagnosis_row["max_road_dist_m"])
    original_p90 = float(diagnosis_row["p90_road_dist_m"])

    snap_displacements = [float(row["snap_distance_m"]) for row in sample_rows if row["snap_applied"] == "True"]
    displacement_mean = sum(snap_displacements) / len(snap_displacements) if snap_displacements else 0.0
    displacement_p95 = percentile(snap_displacements, 0.95)
    displacement_max = max(snap_displacements) if snap_displacements else 0.0

    step_lengths = geometry_step_lengths(candidate_feature)
    max_step = max(step_lengths) if step_lengths else 0.0
    p95_step = percentile(step_lengths, 0.95)

    improves_mean = candidate_mean <= original_mean * 0.7
    improves_p90 = candidate_p90 <= original_p90 * 0.7
    improves_max = candidate_max <= max(25.0, original_max * 0.6)
    displacement_ok = displacement_p95 <= 30.0 and displacement_max <= 40.0
    continuity_ok = max_step <= 35.0 and p95_step <= 20.0
    road_fit_ok = candidate_mean <= 8.0 and candidate_p90 <= 15.0 and candidate_max <= 25.0
    adopt = improves_mean and improves_p90 and improves_max and displacement_ok and continuity_ok and road_fit_ok
    reasons: list[str] = []
    if not improves_mean:
        reasons.append("mean_distance_not_improved_enough")
    if not improves_p90:
        reasons.append("p90_distance_not_improved_enough")
    if not improves_max:
        reasons.append("max_distance_not_improved_enough")
    if not displacement_ok:
        reasons.append("displacement_too_large")
    if not continuity_ok:
        reasons.append("geometry_jump_detected")
    if not road_fit_ok:
        reasons.append("candidate_not_close_enough_to_road")

    return {
        "route_id": route_id,
        "original_mean_road_dist_m": round(original_mean, 3),
        "original_p90_road_dist_m": round(original_p90, 3),
        "original_max_road_dist_m": round(original_max, 3),
        "candidate_mean_road_dist_m": round(candidate_mean, 3),
        "candidate_p90_road_dist_m": round(candidate_p90, 3),
        "candidate_max_road_dist_m": round(candidate_max, 3),
        "snap_displacement_mean_m": round(displacement_mean, 3),
        "snap_displacement_p95_m": round(displacement_p95, 3),
        "snap_displacement_max_m": round(displacement_max, 3),
        "candidate_max_step_m": round(max_step, 3),
        "candidate_p95_step_m": round(p95_step, 3),
        "adopt_candidate": adopt,
        "adoption_reasons": ",".join(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Route-level postprocess for one inspected frame")
    parser.add_argument("--diagnosis", type=Path, default=Path("artifacts/inspection/page_061_061_f06/route_diagnosis.csv"))
    parser.add_argument("--original", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/frame_override/manual_override_routes.geojson"))
    parser.add_argument("--snap-candidate-source", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval/snapped_routes.geojson"))
    parser.add_argument("--road-samples", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval/route_samples.csv"))
    parser.add_argument("--osm-roads", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval/osm_roads.geojson"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postprocess/page_061_061_f06"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    original_by_id = load_geojson_by_route(args.original)
    snapped_by_id = load_geojson_by_route(args.snap_candidate_source)
    diagnosis_rows = load_diagnosis(args.diagnosis)
    diagnosis_by_id = {row["route_id"]: row for row in diagnosis_rows}
    road_sample_rows = route_samples_by_id(args.road_samples)
    roads_geojson = load_geojson(args.osm_roads)
    all_features = list(original_by_id.values()) + list(snapped_by_id.values())
    bbox = (
        min(geometry_bbox(f)[0] for f in all_features),
        min(geometry_bbox(f)[1] for f in all_features),
        max(geometry_bbox(f)[2] for f in all_features),
        max(geometry_bbox(f)[3] for f in all_features),
    )
    ref_lon = (bbox[0] + bbox[2]) / 2.0
    ref_lat = (bbox[1] + bbox[3]) / 2.0
    road_segments = prepare_road_segments(roads_geojson, ref_lon, ref_lat)

    final_features: list[dict[str, Any]] = []
    final_matched_features: list[dict[str, Any]] = []
    original_features: list[dict[str, Any]] = []
    candidate_features: list[dict[str, Any]] = []
    osm_missing_features: list[dict[str, Any]] = []
    manual_features: list[dict[str, Any]] = []
    candidate_assessment_rows: list[dict[str, Any]] = []

    for row in diagnosis_rows:
        route_id = row["route_id"]
        action = row["recommended_action"]
        original = original_by_id[route_id]
        original_features.append(original)

        base_props = dict(original["properties"])
        base_props.update(
            {
                "postprocess_action": action,
                "postprocess_frame_id": "061_f06",
                "diagnosis_pdf_extraction_status": row["pdf_extraction_status"],
                "diagnosis_step3_problem_type": row["step3_problem_type"],
                "diagnosis_likely_issue": row["likely_issue"],
                "diagnosis_mean_road_dist_m": float(row["mean_road_dist_m"]),
                "diagnosis_p90_road_dist_m": float(row["p90_road_dist_m"]),
                "diagnosis_max_road_dist_m": float(row["max_road_dist_m"]),
                "diagnosis_snap_ratio": float(row["snap_ratio"]),
            }
        )

        if action == "accept_as_is":
            base_props["candidate_available"] = False
            feature = make_feature(original["geometry"], base_props)
            final_features.append(feature)
            final_matched_features.append(make_feature(original["geometry"], dict(base_props)))
            continue

        if action == "map_match_candidate":
            candidate = snapped_by_id[route_id]
            assessment = evaluate_candidate(
                route_id,
                original,
                candidate,
                diagnosis_by_id[route_id],
                road_sample_rows.get(route_id, []),
                road_segments,
            )
            candidate_assessment_rows.append(assessment)
            base_props["candidate_available"] = True
            base_props["needs_manual_review"] = True
            base_props["review_reasons"] = append_reason(str(base_props.get("review_reasons") or ""), "map_match_candidate")
            feature = make_feature(original["geometry"], base_props)
            final_features.append(feature)

            matched_props = dict(base_props)
            matched_props["matched_candidate_adopted"] = bool(assessment["adopt_candidate"])
            matched_props["matched_candidate_reason"] = assessment["adoption_reasons"]
            if assessment["adopt_candidate"]:
                matched_props["review_reasons"] = append_reason(str(matched_props.get("review_reasons") or ""), "matched_candidate_adopted")
                final_matched_features.append(make_feature(candidate["geometry"], matched_props))
            else:
                final_matched_features.append(make_feature(original["geometry"], matched_props))

            cand_props = dict(candidate["properties"])
            cand_props.update(
                {
                    "candidate_for_route_id": route_id,
                    "candidate_source": "osm_nearest_road_snap",
                    "postprocess_action": action,
                    "candidate_adoptable": bool(assessment["adopt_candidate"]),
                    "candidate_assessment_reasons": assessment["adoption_reasons"],
                }
            )
            candidate_features.append(make_feature(candidate["geometry"], cand_props))
            continue

        if action == "osm_missing_or_trail":
            base_props["candidate_available"] = False
            base_props["osm_missing_or_trail"] = True
            feature = make_feature(original["geometry"], base_props)
            final_features.append(feature)
            final_matched_features.append(make_feature(original["geometry"], dict(base_props)))
            osm_missing_features.append(make_feature(original["geometry"], dict(base_props)))
            continue

        if action == "manual_edit_needed":
            base_props["candidate_available"] = False
            base_props["needs_manual_review"] = True
            base_props["review_reasons"] = append_reason(str(base_props.get("review_reasons") or ""), "manual_edit_needed")
            feature = make_feature(original["geometry"], base_props)
            final_features.append(feature)
            final_matched_features.append(make_feature(original["geometry"], dict(base_props)))
            manual_features.append(make_feature(original["geometry"], dict(base_props)))
            continue

        base_props["candidate_available"] = False
        feature = make_feature(original["geometry"], base_props)
        final_features.append(feature)
        final_matched_features.append(make_feature(original["geometry"], dict(base_props)))

    write_geojson(args.out_dir / "final_routes.geojson", final_features)
    write_geojson(args.out_dir / "final_routes_matched.geojson", final_matched_features)
    write_geojson(args.out_dir / "original_routes.geojson", original_features)
    write_geojson(args.out_dir / "snap_candidates.geojson", candidate_features)
    write_geojson(args.out_dir / "osm_missing_or_trail.geojson", osm_missing_features)
    write_geojson(args.out_dir / "manual_edit_needed.geojson", manual_features)
    write_report(args.out_dir / "report.md", diagnosis_rows, candidate_assessment_rows)
    (args.out_dir / "candidate_assessment.json").write_text(json.dumps(candidate_assessment_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if candidate_assessment_rows:
        with (args.out_dir / "candidate_assessment.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidate_assessment_rows[0].keys()))
            writer.writeheader()
            writer.writerows(candidate_assessment_rows)

    debug_dir = args.out_dir / "debug"
    ensure_dir(debug_dir)
    for name in [
        "final_routes.geojson",
        "final_routes_matched.geojson",
        "original_routes.geojson",
        "snap_candidates.geojson",
        "osm_missing_or_trail.geojson",
        "manual_edit_needed.geojson",
    ]:
        (debug_dir / name).write_text((args.out_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "route_diagnosis.json").write_text(json.dumps(diagnosis_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "candidate_assessment.json").write_text(json.dumps(candidate_assessment_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "index.html").write_text(build_debug_html(read_mapbox_token()), encoding="utf-8")


if __name__ == "__main__":
    main()
