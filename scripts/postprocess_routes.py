#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
    const state = {{ original: null, final: null, candidates: null, osmMissing: null, manual: null, diagnosis: [] }};
    async function load() {{
      const [origRes, finalRes, candRes, osmRes, manualRes, diagRes] = await Promise.all([
        fetch('original_routes.geojson', {{ cache: 'no-store' }}),
        fetch('final_routes.geojson', {{ cache: 'no-store' }}),
        fetch('snap_candidates.geojson', {{ cache: 'no-store' }}),
        fetch('osm_missing_or_trail.geojson', {{ cache: 'no-store' }}),
        fetch('manual_edit_needed.geojson', {{ cache: 'no-store' }}),
        fetch('route_diagnosis.json', {{ cache: 'no-store' }})
      ]);
      state.original = await origRes.json();
      state.final = await finalRes.json();
      state.candidates = await candRes.json();
      state.osmMissing = await osmRes.json();
      state.manual = await manualRes.json();
      state.diagnosis = await diagRes.json();
      render('all');
    }}
    function ensure() {{
      if (map.getSource('final')) return;
      for (const id of ['original','final','candidates','osmMissing','manual']) {{
        map.addSource(id, {{ type: 'geojson', data: {{ type:'FeatureCollection', features:[] }} }});
      }}
      map.addLayer({{ id:'original-line', type:'line', source:'original', paint:{{ 'line-color':'#9aa1a8', 'line-width':3, 'line-opacity':0.6 }} }});
      map.addLayer({{ id:'final-line', type:'line', source:'final', paint:{{ 'line-color':'#cf1b1b', 'line-width':4 }} }});
      map.addLayer({{ id:'candidate-line', type:'line', source:'candidates', paint:{{ 'line-color':'#0e7a7a', 'line-width':3.5, 'line-dasharray':[1.2,0.8] }} }});
      map.addLayer({{ id:'osm-line', type:'line', source:'osmMissing', paint:{{ 'line-color':'#c88b00', 'line-width':4 }} }});
      map.addLayer({{ id:'manual-line', type:'line', source:'manual', paint:{{ 'line-color':'#6b46c1', 'line-width':4.5 }} }});
    }}
    function setData(mode) {{
      map.getSource('original').setData(mode === 'original' || mode === 'all' ? state.original : {{ type:'FeatureCollection', features:[] }});
      map.getSource('final').setData(mode === 'final' || mode === 'all' ? state.final : {{ type:'FeatureCollection', features:[] }});
      map.getSource('candidates').setData(mode === 'candidates' || mode === 'all' ? state.candidates : {{ type:'FeatureCollection', features:[] }});
      map.getSource('osmMissing').setData(mode === 'all' ? state.osmMissing : {{ type:'FeatureCollection', features:[] }});
      map.getSource('manual').setData(mode === 'all' ? state.manual : {{ type:'FeatureCollection', features:[] }});
    }}
    function renderSummary() {{
      document.getElementById('summary').innerHTML = state.diagnosis.map(row => `
        <div class="item">
          <strong>${{row.route_id}}</strong><br>
          action=${{row.recommended_action}}<br>
          likely_issue=${{row.likely_issue}}<br>
          mean=${{row.mean_road_dist_m}}m / max=${{row.max_road_dist_m}}m
        </div>
      `).join('');
    }}
    function fitBounds() {{
      const bounds = new mapboxgl.LngLatBounds();
      for (const collection of [state.final, state.original, state.candidates]) {{
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


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["recommended_action"]] = counts.get(row["recommended_action"], 0) + 1
    lines = [
        "# Postprocess Report",
        "",
        "- Frame: `061_f06`",
        f"- Route count: `{len(rows)}`",
        "",
        "## Action Counts",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: `{counts[key]}`")
    lines.extend(
        [
            "",
            "## Notes",
            "- `map_match_candidate` は最終採用ではなく candidate です。",
            "- `osm_missing_or_trail` は元線を保持し、snap していません。",
            "- `manual_edit_needed` は元線を保持し、needs_manual_review を維持しています。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Route-level postprocess for one inspected frame")
    parser.add_argument("--diagnosis", type=Path, default=Path("artifacts/inspection/page_061_061_f06/route_diagnosis.csv"))
    parser.add_argument("--original", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/frame_override/manual_override_routes.geojson"))
    parser.add_argument("--snap-candidate-source", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval/snapped_routes.geojson"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postprocess/page_061_061_f06"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    original_by_id = load_geojson_by_route(args.original)
    snapped_by_id = load_geojson_by_route(args.snap_candidate_source)
    diagnosis_rows = load_diagnosis(args.diagnosis)

    final_features: list[dict[str, Any]] = []
    original_features: list[dict[str, Any]] = []
    candidate_features: list[dict[str, Any]] = []
    osm_missing_features: list[dict[str, Any]] = []
    manual_features: list[dict[str, Any]] = []

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
            final_features.append(make_feature(original["geometry"], base_props))
            continue

        if action == "map_match_candidate":
            candidate = snapped_by_id[route_id]
            base_props["candidate_available"] = True
            base_props["needs_manual_review"] = True
            base_props["review_reasons"] = append_reason(str(base_props.get("review_reasons") or ""), "map_match_candidate")
            final_features.append(make_feature(original["geometry"], base_props))

            cand_props = dict(candidate["properties"])
            cand_props.update(
                {
                    "candidate_for_route_id": route_id,
                    "candidate_source": "osm_nearest_road_snap",
                    "postprocess_action": action,
                }
            )
            candidate_features.append(make_feature(candidate["geometry"], cand_props))
            continue

        if action == "osm_missing_or_trail":
            base_props["candidate_available"] = False
            base_props["osm_missing_or_trail"] = True
            final_features.append(make_feature(original["geometry"], base_props))
            osm_missing_features.append(make_feature(original["geometry"], dict(base_props)))
            continue

        if action == "manual_edit_needed":
            base_props["candidate_available"] = False
            base_props["needs_manual_review"] = True
            base_props["review_reasons"] = append_reason(str(base_props.get("review_reasons") or ""), "manual_edit_needed")
            final_features.append(make_feature(original["geometry"], base_props))
            manual_features.append(make_feature(original["geometry"], dict(base_props)))
            continue

        base_props["candidate_available"] = False
        final_features.append(make_feature(original["geometry"], base_props))

    write_geojson(args.out_dir / "final_routes.geojson", final_features)
    write_geojson(args.out_dir / "original_routes.geojson", original_features)
    write_geojson(args.out_dir / "snap_candidates.geojson", candidate_features)
    write_geojson(args.out_dir / "osm_missing_or_trail.geojson", osm_missing_features)
    write_geojson(args.out_dir / "manual_edit_needed.geojson", manual_features)
    write_report(args.out_dir / "report.md", diagnosis_rows)

    debug_dir = args.out_dir / "debug"
    ensure_dir(debug_dir)
    for name in [
        "final_routes.geojson",
        "original_routes.geojson",
        "snap_candidates.geojson",
        "osm_missing_or_trail.geojson",
        "manual_edit_needed.geojson",
    ]:
        (debug_dir / name).write_text((args.out_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "route_diagnosis.json").write_text(json.dumps(diagnosis_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "index.html").write_text(build_debug_html(read_mapbox_token()), encoding="utf-8")


if __name__ == "__main__":
    main()
