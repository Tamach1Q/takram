#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw


TARGET_RED = (237 / 255.0, 28 / 255.0, 36 / 255.0)
DEFAULT_L1_THRESHOLD = 0.08
DEFAULT_PREVIEW_PAGES = 5
PRIMARY_ROUTE_CLASS = "route_candidate_solid_main"
DASHED_NONROUTE_CLASSES = {"red_dashed_nonroute", "annotation_dashed"}
ANNOTATION_CLASSES = {"filled_symbol_or_legend", "small_symbol_or_label", "legend_like", "red_annotation_solid", "unknown_red"}


@dataclass
class DrawMetrics:
    length_pt: float
    segment_count: int
    curve_count: int
    closed_count: int


def color_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        comp = float(value)
        return [comp, comp, comp]
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        return [float(part) for part in value[:3]]
    return None


def l1_distance(color: list[float] | None, target: tuple[float, float, float]) -> float | None:
    if color is None:
        return None
    if len(color) < 3:
        return None
    return sum(abs(color[idx] - target[idx]) for idx in range(3))


def point_tuple(point: Any) -> tuple[float, float]:
    return (float(point.x), float(point.y))


def sample_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 16,
) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for step in range(steps + 1):
        t = step / steps
        mt = 1.0 - t
        x = (
            (mt ** 3) * p0[0]
            + 3 * (mt ** 2) * t * p1[0]
            + 3 * mt * (t ** 2) * p2[0]
            + (t ** 3) * p3[0]
        )
        y = (
            (mt ** 3) * p0[1]
            + 3 * (mt ** 2) * t * p1[1]
            + 3 * mt * (t ** 2) * p2[1]
            + (t ** 3) * p3[1]
        )
        sampled.append((x, y))
    return sampled


def polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.dist(start, end)
    return total


def normalize_dash(dashes: Any) -> str:
    if dashes in (None, "", "[] 0"):
        return "solid"
    return str(dashes).strip()


def metrics_from_items(items: list[Any]) -> DrawMetrics:
    total_length = 0.0
    segment_count = 0
    curve_count = 0
    closed_count = 0

    for item in items:
        operator = item[0]
        if operator == "l":
            start = point_tuple(item[1])
            end = point_tuple(item[2])
            total_length += math.dist(start, end)
            segment_count += 1
        elif operator == "c":
            p0 = point_tuple(item[1])
            p1 = point_tuple(item[2])
            p2 = point_tuple(item[3])
            p3 = point_tuple(item[4])
            total_length += polyline_length(sample_cubic_bezier(p0, p1, p2, p3))
            curve_count += 1
            segment_count += 1
        elif operator == "re":
            rect = item[1]
            total_length += 2.0 * (float(rect.width) + float(rect.height))
            closed_count += 1
            segment_count += 4
        elif operator == "qu":
            quad = item[1]
            points = [point_tuple(quad.ul), point_tuple(quad.ur), point_tuple(quad.lr), point_tuple(quad.ll), point_tuple(quad.ul)]
            total_length += polyline_length(points)
            closed_count += 1
            segment_count += 4
        elif operator == "h":
            closed_count += 1

    return DrawMetrics(
        length_pt=total_length,
        segment_count=segment_count,
        curve_count=curve_count,
        closed_count=closed_count,
    )


def polylines_from_items(items: list[Any]) -> list[list[tuple[float, float]]]:
    polylines: list[list[tuple[float, float]]] = []
    current_path_start: tuple[float, float] | None = None
    current_point: tuple[float, float] | None = None

    for item in items:
        operator = item[0]
        if operator == "l":
            start = point_tuple(item[1])
            end = point_tuple(item[2])
            polylines.append([start, end])
            if current_path_start is None:
                current_path_start = start
            current_point = end
        elif operator == "c":
            p0 = point_tuple(item[1])
            p1 = point_tuple(item[2])
            p2 = point_tuple(item[3])
            p3 = point_tuple(item[4])
            polylines.append(sample_cubic_bezier(p0, p1, p2, p3))
            if current_path_start is None:
                current_path_start = p0
            current_point = p3
        elif operator == "re":
            rect = item[1]
            points = [
                (float(rect.x0), float(rect.y0)),
                (float(rect.x1), float(rect.y0)),
                (float(rect.x1), float(rect.y1)),
                (float(rect.x0), float(rect.y1)),
                (float(rect.x0), float(rect.y0)),
            ]
            polylines.append(points)
            current_path_start = None
            current_point = None
        elif operator == "qu":
            quad = item[1]
            points = [
                point_tuple(quad.ul),
                point_tuple(quad.ur),
                point_tuple(quad.lr),
                point_tuple(quad.ll),
                point_tuple(quad.ul),
            ]
            polylines.append(points)
            current_path_start = None
            current_point = None
        elif operator == "h":
            if current_path_start is not None and current_point is not None and current_point != current_path_start:
                polylines.append([current_point, current_path_start])
            current_path_start = None
            current_point = None

    return polylines


def classify_red_object(row: dict[str, Any]) -> str:
    dash = row["dashes"]
    width = row["width_pt"]
    item_type = row["draw_type"]
    rect_w = row["bbox_w_pt"]
    rect_h = row["bbox_h_pt"]
    path_len = row["path_length_pt"]
    fill_distance = row["fill_l1_distance"]
    closed_count = row["closed_count"]
    segment_count = row["segment_count"]
    boxy_perimeter = (2.0 * (rect_w + rect_h)) if rect_w > 0 and rect_h > 0 else 0.0
    box_like = closed_count > 0 and rect_w >= 18 and rect_h >= 18 and path_len <= boxy_perimeter + 12.0

    if item_type == "fs" or fill_distance is not None:
        return "filled_symbol_or_legend"
    if rect_w < 14 and rect_h < 14 and path_len < 40:
        return "small_symbol_or_label"
    if dash != "solid":
        if 0.8 <= width <= 2.4 and path_len >= 20:
            return "red_dashed_nonroute"
        return "annotation_dashed"
    if rect_w > 120 and rect_h < 40 and path_len < 300:
        return "legend_like"
    if box_like:
        return "red_annotation_solid"
    if dash == "solid" and 0.8 <= width <= 2.4 and path_len >= 15 and segment_count >= 1:
        return PRIMARY_ROUTE_CLASS
    if path_len >= 10:
        return "red_annotation_solid"
    return "unknown_red"


def overlay_rect(
    draw: ImageDraw.ImageDraw,
    rect: tuple[float, float, float, float],
    scale: float,
    color: str,
    width: int = 3,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=color, width=width)


def overlay_polylines(
    draw: ImageDraw.ImageDraw,
    polylines: list[list[tuple[float, float]]],
    scale: float,
    color: str,
    width: int,
) -> None:
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        scaled = [(x * scale, y * scale) for x, y in polyline]
        draw.line(scaled, fill=color, width=width, joint="curve")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, fitz.Rect):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    if isinstance(value, fitz.Point):
        return [float(value.x), float(value.y)]
    if isinstance(value, fitz.Quad):
        return {
            "ul": [float(value.ul.x), float(value.ul.y)],
            "ur": [float(value.ur.x), float(value.ur.y)],
            "ll": [float(value.ll.x), float(value.ll.y)],
            "lr": [float(value.lr.x), float(value.lr.y)],
        }
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def scan_pdf(pdf_path: Path, out_dir: Path, l1_threshold: float, preview_pages: int) -> dict[str, Any]:
    ensure_dir(out_dir)
    previews_dir = out_dir / "previews"
    ensure_dir(previews_dir)

    red_objects_csv = out_dir / "red_objects.csv"
    page_summary_csv = out_dir / "page_red_summary.csv"
    log_json = out_dir / "extraction_log.json"
    report_md = out_dir / "step1_report.md"

    doc = fitz.open(pdf_path)

    page_rows: list[dict[str, Any]] = []
    red_rows: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        drawings = page.get_drawings()
        dash_counter: Counter[str] = Counter()
        width_counter: Counter[str] = Counter()
        page_red_count = 0
        page_red_length = 0.0
        class_counter: Counter[str] = Counter()

        for draw_index, drawing in enumerate(drawings):
            stroke = color_to_list(drawing.get("color"))
            fill = color_to_list(drawing.get("fill"))
            stroke_l1 = l1_distance(stroke, TARGET_RED)
            fill_l1 = l1_distance(fill, TARGET_RED)
            is_red = (
                stroke_l1 is not None and stroke_l1 <= l1_threshold
            ) or (
                fill_l1 is not None and fill_l1 <= l1_threshold
            )

            if not is_red:
                continue

            metrics = metrics_from_items(drawing.get("items", []))
            rect = drawing.get("rect") or fitz.Rect(0, 0, 0, 0)
            row = {
                "page_no": page_index + 1,
                "draw_index": draw_index,
                "draw_type": drawing.get("type"),
                "stroke_rgb": json.dumps(stroke, ensure_ascii=False),
                "fill_rgb": json.dumps(fill, ensure_ascii=False),
                "stroke_l1_distance": round(stroke_l1, 6) if stroke_l1 is not None else None,
                "fill_l1_distance": round(fill_l1, 6) if fill_l1 is not None else None,
                "width_pt": float(drawing.get("width") or 0.0),
                "dashes": normalize_dash(drawing.get("dashes")),
                "bbox_x0_pt": float(rect.x0),
                "bbox_y0_pt": float(rect.y0),
                "bbox_x1_pt": float(rect.x1),
                "bbox_y1_pt": float(rect.y1),
                "bbox_w_pt": float(rect.width),
                "bbox_h_pt": float(rect.height),
                "path_length_pt": round(metrics.length_pt, 3),
                "segment_count": metrics.segment_count,
                "curve_count": metrics.curve_count,
                "closed_count": metrics.closed_count,
            }
            row["classification"] = classify_red_object(row)

            dash_counter[row["dashes"]] += 1
            width_counter[f"{row['width_pt']:.2f}"] += 1
            class_counter[row["classification"]] += 1
            page_red_count += 1
            page_red_length += row["path_length_pt"]
            red_rows.append(row | {"raw_drawing": to_jsonable(drawing)})

        page_rows.append(
            {
                "page_no": page_index + 1,
                "page_width_pt": round(float(page.rect.width), 3),
                "page_height_pt": round(float(page.rect.height), 3),
                "red_object_count": page_red_count,
                "red_total_path_length_pt": round(page_red_length, 3),
                "dash_patterns": json.dumps(dash_counter, ensure_ascii=False, sort_keys=True),
                "width_distribution_pt": json.dumps(width_counter, ensure_ascii=False, sort_keys=True),
                "classification_counts": json.dumps(class_counter, ensure_ascii=False, sort_keys=True),
            }
        )

    red_rows_sorted = sorted(
        red_rows,
        key=lambda row: (
            row["classification"] != PRIMARY_ROUTE_CLASS,
            -row["path_length_pt"],
            row["page_no"],
            row["draw_index"],
        ),
    )
    preview_target_pages = [
        row["page_no"]
        for row in sorted(
            page_rows,
            key=lambda row: (row["red_total_path_length_pt"], row["red_object_count"]),
            reverse=True,
        )[:preview_pages]
    ]

    with red_objects_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in red_rows_sorted[0].keys() if key != "raw_drawing"] if red_rows_sorted else [
            "page_no",
            "draw_index",
            "draw_type",
            "stroke_rgb",
            "fill_rgb",
            "stroke_l1_distance",
            "fill_l1_distance",
            "width_pt",
            "dashes",
            "bbox_x0_pt",
            "bbox_y0_pt",
            "bbox_x1_pt",
            "bbox_y1_pt",
            "bbox_w_pt",
            "bbox_h_pt",
            "path_length_pt",
            "segment_count",
            "curve_count",
            "closed_count",
            "classification",
        ])
        writer.writeheader()
        for row in red_rows_sorted:
            writer.writerow({key: value for key, value in row.items() if key != "raw_drawing"})

    with page_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(page_rows[0].keys()) if page_rows else [
            "page_no",
            "page_width_pt",
            "page_height_pt",
            "red_object_count",
            "red_total_path_length_pt",
            "dash_patterns",
            "width_distribution_pt",
            "classification_counts",
        ])
        writer.writeheader()
        writer.writerows(page_rows)

    preview_log: list[dict[str, Any]] = []
    for page_no in preview_target_pages:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image, "RGBA")
        scale = 2.0
        page_objects = [row for row in red_rows_sorted if row["page_no"] == page_no]
        drawings = page.get_drawings()

        for row in page_objects:
            drawing = drawings[row["draw_index"]]
            polylines = polylines_from_items(drawing.get("items", []))
            rect = (
                row["bbox_x0_pt"],
                row["bbox_y0_pt"],
                row["bbox_x1_pt"],
                row["bbox_y1_pt"],
            )
            if row["classification"] == PRIMARY_ROUTE_CLASS:
                overlay_polylines(draw, polylines, scale, "#DC2626", width=5)
            elif row["classification"] in DASHED_NONROUTE_CLASSES:
                overlay_polylines(draw, polylines, scale, "#2563EB", width=4)
            elif row["classification"] in ANNOTATION_CLASSES:
                overlay_polylines(draw, polylines, scale, "#6B7280", width=2)
                if row["classification"] in {"filled_symbol_or_legend", "legend_like", "red_annotation_solid"} and row["bbox_w_pt"] * row["bbox_h_pt"] > 1200:
                    overlay_rect(draw, rect, scale, "#6B7280", width=1)
            else:
                overlay_polylines(draw, polylines, scale, "#FF4D6D", width=2)
                if not polylines:
                    overlay_rect(draw, rect, scale, "#FF4D6D", width=2)

        preview_path = previews_dir / f"page_{page_no:03d}.png"
        image.save(preview_path)
        preview_log.append(
            {
                "page_no": page_no,
                "preview_path": str(preview_path),
                "red_object_count": sum(1 for row in page_objects),
            }
        )

    class_counts = Counter(row["classification"] for row in red_rows_sorted)
    dash_counts = Counter(row["dashes"] for row in red_rows_sorted)
    width_counts = Counter(f"{row['width_pt']:.2f}" for row in red_rows_sorted)

    result = {
        "pdf_path": str(pdf_path),
        "page_count": doc.page_count,
        "target_red": TARGET_RED,
        "l1_threshold": l1_threshold,
        "preview_pages": preview_target_pages,
        "red_object_count": len(red_rows_sorted),
        "classification_counts": class_counts,
        "dash_counts": dash_counts,
        "width_counts": width_counts,
        "outputs": {
            "red_objects_csv": str(red_objects_csv),
            "page_summary_csv": str(page_summary_csv),
            "report_md": str(report_md),
            "previews_dir": str(previews_dir),
        },
    }

    with log_json.open("w", encoding="utf-8") as handle:
        json.dump(
            result | {
                "previews": preview_log,
                "sample_red_objects": [
                    {key: value for key, value in row.items() if key != "raw_drawing"}
                    for row in red_rows_sorted[:25]
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=lambda value: dict(value),
        )

    report_lines = [
        "# Step 1 Report",
        "",
        f"- PDF: `{pdf_path.name}`",
        f"- Pages scanned: `{doc.page_count}`",
        f"- Red-object threshold (L1): `{l1_threshold}`",
        f"- Red objects found: `{len(red_rows_sorted)}`",
        "",
        "## Representative Pages",
    ]
    for preview in preview_log:
        report_lines.append(
            f"- Page {preview['page_no']}: `{Path(preview['preview_path']).name}` ({preview['red_object_count']} red objects)"
        )

    report_lines.extend(
        [
            "",
            "## Heuristic Split",
            f"- Solid route candidates: `{class_counts.get(PRIMARY_ROUTE_CLASS, 0)}`",
            f"- Excluded dashed non-route: `{class_counts.get('red_dashed_nonroute', 0)}`",
            f"- Dashed annotations: `{class_counts.get('annotation_dashed', 0)}`",
            f"- Filled symbols / legends: `{class_counts.get('filled_symbol_or_legend', 0)}`",
            f"- Small symbols / labels: `{class_counts.get('small_symbol_or_label', 0)}`",
            f"- Solid red annotations: `{class_counts.get('red_annotation_solid', 0)}`",
            f"- Unknown red: `{class_counts.get('unknown_red', 0)}`",
            "",
            "## Dominant Dash Patterns",
        ]
    )
    for dash, count in dash_counts.most_common(10):
        report_lines.append(f"- `{dash}`: `{count}`")

    report_lines.extend(["", "## Dominant Widths"])
    for width, count in width_counts.most_common(10):
        report_lines.append(f"- `{width} pt`: `{count}`")

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan red vector objects in a PDF map and generate Step 1 artifacts.")
    parser.add_argument("pdf", type=Path, help="Path to the target PDF")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/step1"), help="Output directory")
    parser.add_argument("--l1-threshold", type=float, default=DEFAULT_L1_THRESHOLD, help="L1 distance threshold for red matching")
    parser.add_argument("--preview-pages", type=int, default=DEFAULT_PREVIEW_PAGES, help="Number of representative preview pages")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = scan_pdf(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        l1_threshold=args.l1_threshold,
        preview_pages=args.preview_pages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: dict(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
