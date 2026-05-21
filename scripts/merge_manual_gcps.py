#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AUTO_FIELDNAMES = [
    "gcp_id",
    "page_no",
    "frame_id",
    "temple_group",
    "temple_no",
    "gazetteer_name_full",
    "gazetteer_name_short",
    "gazetteer_reading",
    "source_name_text",
    "source_number_text",
    "source_reading_text",
    "latitude",
    "longitude",
    "pdf_label_x_pt",
    "pdf_label_y_pt",
    "pdf_anchor_x_pt",
    "pdf_anchor_y_pt",
    "marker_found",
    "snap_distance_pt",
    "marker_score",
    "name_similarity",
    "confidence",
    "needs_manual_review",
    "review_reasons",
    "source_kind",
    "gazetteer_source",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_page_heights(path: Path) -> dict[int, float]:
    heights: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("page_no") or not row.get("page_height_pt"):
                continue
            heights[int(row["page_no"])] = float(row["page_height_pt"])
    return heights


def read_auto_rows(path: Path, page_heights: dict[int, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            page_no = int(row["page_no"])
            merged = dict(row)
            merged["page_height_pt"] = f"{page_heights[page_no]:.3f}" if page_no in page_heights else ""
            merged["source"] = "auto"
            merged["role"] = "auto_gcp"
            rows.append(merged)
    return rows


def canonical_manual_filename(page_no: int, frame_id: str) -> str:
    return f"page_{page_no:03d}_{frame_id}.json"


def parse_manual_file(path: Path, page_heights: dict[int, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_rows: list[dict[str, Any]] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [
            {
                "file_name": path.name,
                "page_no": "",
                "frame_id": "",
                "status": "skipped_invalid_json",
                "reason": str(exc),
                "frame_anchor_count": 0,
                "imported_count": 0,
            }
        ]

    page_no = int(data.get("page_no") or 0)
    frame_id = str(data.get("frame_id") or "")
    expected_name = canonical_manual_filename(page_no, frame_id) if page_no and frame_id else ""
    anchors = [row for row in data.get("gcps", []) if row.get("role") == "frame_anchor"]
    if path.name != expected_name:
        report_rows.append(
            {
                "file_name": path.name,
                "page_no": page_no,
                "frame_id": frame_id,
                "status": "skipped_variant",
                "reason": f"expected {expected_name}",
                "frame_anchor_count": len(anchors),
                "imported_count": 0,
            }
        )
        return [], report_rows
    if not anchors:
        report_rows.append(
            {
                "file_name": path.name,
                "page_no": page_no,
                "frame_id": frame_id,
                "status": "skipped_no_frame_anchor",
                "reason": "frame_anchor がない",
                "frame_anchor_count": 0,
                "imported_count": 0,
            }
        )
        return [], report_rows

    page_height_pt = page_heights.get(page_no)
    if page_height_pt is None:
        page = data.get("page", {})
        rect = page.get("rect_pt", {})
        if "y0" in rect and "y1" in rect:
            page_height_pt = float(rect["y1"]) - float(rect["y0"])
    if page_height_pt is None:
        report_rows.append(
            {
                "file_name": path.name,
                "page_no": page_no,
                "frame_id": frame_id,
                "status": "skipped_missing_page_height",
                "reason": "page_height_pt が取得できない",
                "frame_anchor_count": len(anchors),
                "imported_count": 0,
            }
        )
        return [], report_rows

    manual_rows: list[dict[str, Any]] = []
    for index, row in enumerate(anchors, start=1):
        manual_rows.append(
            {
                "gcp_id": f"manual_{page_no}_{frame_id}_{index}",
                "page_no": str(page_no),
                "frame_id": frame_id,
                "temple_group": "manual",
                "temple_no": str(index),
                "gazetteer_name_full": "",
                "gazetteer_name_short": "",
                "gazetteer_reading": "",
                "source_name_text": str(row.get("name", "")),
                "source_number_text": "",
                "source_reading_text": "",
                "latitude": str(float(row["latitude"])),
                "longitude": str(float(row["longitude"])),
                "pdf_label_x_pt": str(float(row["pdf_x"])),
                "pdf_label_y_pt": str(float(row.get("raw_pdf_y_top_left", row["pdf_y"]))),
                "pdf_anchor_x_pt": str(float(row["pdf_x"])),
                "pdf_anchor_y_pt": str(float(row.get("raw_pdf_y_top_left", row["pdf_y"]))),
                "marker_found": "True",
                "snap_distance_pt": "0.0",
                "marker_score": "1.0",
                "name_similarity": "1.0",
                "confidence": "1.0",
                "needs_manual_review": "False",
                "review_reasons": "manual_frame_anchor",
                "source_kind": "manual_frame_anchor",
                "gazetteer_source": "",
                "page_height_pt": f"{float(page_height_pt):.3f}",
                "source": "manual",
                "role": "frame_anchor",
            }
        )

    report_rows.append(
        {
            "file_name": path.name,
            "page_no": page_no,
            "frame_id": frame_id,
            "status": "imported",
            "reason": "manual_frame_anchor imported",
            "frame_anchor_count": len(anchors),
            "imported_count": len(manual_rows),
        }
    )
    return manual_rows, report_rows


def merge_rows(auto_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manual_frames = {(int(row["page_no"]), row["frame_id"]) for row in manual_rows}
    merged: list[dict[str, Any]] = []
    for row in auto_rows:
        key = (int(row["page_no"]), row["frame_id"])
        if key in manual_frames:
            continue
        merged.append(row)
    merged.extend(manual_rows)
    merged.sort(key=lambda row: (int(row["page_no"]), row["frame_id"], row["source"], row["gcp_id"]))
    return merged


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge manual GCP JSON into Step 5 GCP input")
    parser.add_argument("--auto-gcps", type=Path, default=Path("artifacts/step4/gcp_candidates.csv"))
    parser.add_argument("--manual-dir", type=Path, default=Path("data/manual_gcps"))
    parser.add_argument("--page-metadata", type=Path, default=Path("artifacts/step1/page_red_summary.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/step5/gcp_candidates_merged.csv"))
    parser.add_argument("--report-csv", type=Path, default=Path("artifacts/step5/manual_gcp_import_report.csv"))
    args = parser.parse_args()

    page_heights = load_page_heights(args.page_metadata)
    auto_rows = read_auto_rows(args.auto_gcps, page_heights)

    manual_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for path in sorted(args.manual_dir.glob("*.json")):
        rows, reports = parse_manual_file(path, page_heights)
        manual_rows.extend(rows)
        report_rows.extend(reports)

    merged_rows = merge_rows(auto_rows, manual_rows)
    merged_fields = AUTO_FIELDNAMES + ["page_height_pt", "source", "role"]
    report_fields = ["file_name", "page_no", "frame_id", "status", "reason", "frame_anchor_count", "imported_count"]

    write_csv(args.out_csv, merged_rows, merged_fields)
    write_csv(args.report_csv, report_rows, report_fields)


if __name__ == "__main__":
    main()
