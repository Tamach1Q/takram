#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameKey:
    page_no: int
    frame_id: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def geometry_length(geometry: dict[str, Any]) -> float:
    geom_type = geometry["type"]
    if geom_type == "LineString":
        lines = [geometry["coordinates"]]
    elif geom_type == "MultiLineString":
        lines = geometry["coordinates"]
    else:
        return 0.0
    total = 0.0
    for line in lines:
        for start, end in zip(line, line[1:]):
            total += math.dist(start, end)
    return total


def load_route_features(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    features = read_geojson(path)["features"]
    by_id = {feature["properties"]["route_id"]: feature for feature in features}
    return features, by_id


def load_transform_status(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {row["route_id"]: row for row in csv.DictReader(handle)}


def load_frame_models(path: Path) -> dict[FrameKey, dict[str, Any]]:
    rows: dict[FrameKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = FrameKey(page_no=to_int(row["page_no"]), frame_id=row["frame_id"])
            rows[key] = row
    return rows


def manual_file_priority(path: Path, page_no: int, frame_id: str, frame_anchor_count: int) -> tuple[int, int, int, str]:
    canonical_name = f"page_{page_no:03d}_{frame_id}.json"
    is_canonical = int(path.name == canonical_name)
    has_gcps = int(frame_anchor_count > 0)
    return (has_gcps, is_canonical, frame_anchor_count, path.name)


def load_manual_gcps(path: Path) -> dict[FrameKey, dict[str, Any]]:
    selected: dict[FrameKey, dict[str, Any]] = {}
    best_priority: dict[FrameKey, tuple[int, int, int, str]] = {}
    for json_path in sorted(path.glob("*.json")):
        if json_path.name.endswith(".template.json"):
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        page_no = to_int(payload.get("page_no"))
        frame_id = str(payload.get("frame_id") or "").strip()
        if not page_no or not frame_id:
            continue
        gcps = payload.get("gcps") or []
        frame_anchors = [gcp for gcp in gcps if gcp.get("role") == "frame_anchor"]
        key = FrameKey(page_no=page_no, frame_id=frame_id)
        priority = manual_file_priority(json_path, page_no, frame_id, len(frame_anchors))
        if key not in selected or priority > best_priority[key]:
            selected[key] = {
                "page_no": page_no,
                "frame_id": frame_id,
                "manual_gcp_count": len(frame_anchors),
                "has_manual_gcp": len(frame_anchors) > 0,
                "source_file": json_path.name,
            }
            best_priority[key] = priority
    return selected


def is_untransformed(status_row: dict[str, Any] | None) -> bool:
    if not status_row:
        return True
    if not str(status_row.get("transform_model") or "").strip():
        return True
    reasons = str(status_row.get("review_reasons") or "")
    return "route_not_transformed" in reasons


def selected_model_label(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    model = str(row.get("transform_model") or "").strip()
    y_mode = str(row.get("y_mode") or "").strip()
    if model and y_mode:
        return f"{model} / {y_mode}"
    return model


def quality_rank(status: str) -> int:
    normalized = status.strip().lower()
    if normalized == "fail":
        return 3
    if normalized == "review":
        return 2
    if not normalized:
        return 1
    return 0


def build_priority(
    row: dict[str, Any],
    max_total_length: float,
) -> tuple[float, list[str]]:
    untransformed = int(row["untransformed_route_count"])
    total_length = float(row["total_route_length"])
    has_manual_gcp = row["has_manual_gcp"] == "True"
    selected_model = row["selected_model"]
    quality_status = row["quality_status"].strip().lower()

    score = 0.0
    reasons: list[str] = []

    if untransformed:
        score += untransformed * 12.0
        reasons.append(f"untransformed_routes={untransformed}")

    if max_total_length > 0:
        length_score = (total_length / max_total_length) * 25.0
        score += length_score
        if total_length >= max_total_length * 0.5:
            reasons.append("long_total_route_length")

    if not selected_model:
        score += 30.0
        reasons.append("frame_model_missing")

    if quality_status == "fail":
        score += 25.0
        reasons.append("auto_gcp_fail")
    elif quality_status == "review":
        score += 15.0
        reasons.append("auto_gcp_review")

    if not has_manual_gcp:
        score += 20.0
        reasons.append("no_manual_gcp")

    if not reasons:
        reasons.append("low_priority")

    return round(score, 3), reasons


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with_manual = sum(1 for row in rows if row["has_manual_gcp"] == "True")
    model_missing = sum(1 for row in rows if not row["selected_model"])
    review_or_fail = sum(1 for row in rows if row["quality_status"] in {"review", "fail"})
    top_rows = rows[:20]

    lines = [
        "# Manual Review Queue",
        "",
        f"- Frame count: `{len(rows)}`",
        f"- Frames with manual GCP: `{with_manual}`",
        f"- Frames without selected model: `{model_missing}`",
        f"- Frames with review/fail model: `{review_or_fail}`",
        "- `total_route_length` は `artifacts/step3/merged_routes.geojson` の `pdf_debug_local` 長です。",
        "",
        "## Top 20",
        "",
        "| Rank | Page | Frame | Routes | Length | Untransformed | Manual GCP | Model | Quality | LOOCV RMSE | Priority | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{index} | {row['page_no']} | {row['frame_id']} | {row['route_count']} | {row['total_route_length']} | "
            f"{row['untransformed_route_count']} | {row['manual_gcp_count']} | {row['selected_model'] or '-'} | "
            f"{row['quality_status'] or '-'} | {row['loocv_rmse_m'] or '-'} | {row['priority_score']} | {row['priority_reason']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frame-level manual GCP review queue")
    parser.add_argument("--routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--transform-status", type=Path, default=Path("artifacts/step5/route_transform_status.csv"))
    parser.add_argument("--frame-models", type=Path, default=Path("artifacts/step5/frame_models.csv"))
    parser.add_argument("--manual-gcp-dir", type=Path, default=Path("data/manual_gcps"))
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/step5/manual_review_queue.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/step5/manual_review_queue.md"))
    args = parser.parse_args()

    ensure_dir(args.out_csv.parent)

    route_features, _ = load_route_features(args.routes)
    status_by_route = load_transform_status(args.transform_status)
    frame_models = load_frame_models(args.frame_models)
    manual_gcps = load_manual_gcps(args.manual_gcp_dir)

    grouped: dict[FrameKey, dict[str, Any]] = {}
    skipped_without_frame = 0
    for feature in route_features:
        props = feature["properties"]
        page_no = to_int(props.get("page_no"))
        frame_id = str(props.get("frame_id") or "").strip()
        if not page_no or not frame_id:
            skipped_without_frame += 1
            continue
        key = FrameKey(page_no=page_no, frame_id=frame_id)
        bucket = grouped.setdefault(
            key,
            {
                "page_no": page_no,
                "frame_id": frame_id,
                "route_count": 0,
                "total_route_length": 0.0,
                "untransformed_route_count": 0,
            },
        )
        bucket["route_count"] += 1
        bucket["total_route_length"] += geometry_length(feature["geometry"])
        route_id = props["route_id"]
        if is_untransformed(status_by_route.get(route_id)):
            bucket["untransformed_route_count"] += 1

    max_total_length = max((row["total_route_length"] for row in grouped.values()), default=0.0)
    rows: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        model_row = frame_models.get(key)
        manual_row = manual_gcps.get(
            key,
            {
                "has_manual_gcp": False,
                "manual_gcp_count": 0,
                "source_file": "",
            },
        )
        row = {
            "page_no": bucket["page_no"],
            "frame_id": bucket["frame_id"],
            "route_count": bucket["route_count"],
            "total_route_length": round(bucket["total_route_length"], 3),
            "untransformed_route_count": bucket["untransformed_route_count"],
            "has_manual_gcp": str(bool(manual_row["has_manual_gcp"])),
            "manual_gcp_count": manual_row["manual_gcp_count"],
            "selected_model": selected_model_label(model_row),
            "quality_status": (model_row or {}).get("quality_status", ""),
            "loocv_rmse_m": (model_row or {}).get("loocv_rmse_m", ""),
            "manual_gcp_source_file": manual_row.get("source_file", ""),
        }
        priority_score, reasons = build_priority(row, max_total_length=max_total_length)
        row["priority_score"] = priority_score
        row["priority_reason"] = ",".join(reasons)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["untransformed_route_count"]),
            -float(row["total_route_length"]),
            -quality_rank(str(row["quality_status"])),
            int(row["page_no"]),
            row["frame_id"],
        )
    )

    fieldnames = [
        "page_no",
        "frame_id",
        "route_count",
        "total_route_length",
        "untransformed_route_count",
        "has_manual_gcp",
        "manual_gcp_count",
        "selected_model",
        "quality_status",
        "loocv_rmse_m",
        "priority_score",
        "priority_reason",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    write_markdown(args.out_md, rows)

    if skipped_without_frame:
        print(f"Skipped routes without frame_id: {skipped_without_frame}")
    print(f"Wrote {len(rows)} frame rows to {args.out_csv}")


if __name__ == "__main__":
    main()
