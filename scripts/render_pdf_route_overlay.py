#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageColor, ImageDraw, ImageFont


MAPBOX_SCALE = 0.01
FULL_SCALE = 1.25
FRAME_MARGIN_PT = 12.0
RED_RATIO_OK = 0.55


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def synthetic_to_pdf(coord: list[float]) -> tuple[float, float]:
    return (coord[0] / MAPBOX_SCALE, -coord[1] / MAPBOX_SCALE)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_page_meta(path: Path, page_no: int) -> dict[str, Any]:
    for row in read_csv_rows(path):
        if int(row["page_no"]) == page_no:
            return row
    raise SystemExit(f"page metadata が見つかりません: page {page_no}")


def load_frame_row(path: Path, frame_id: str) -> dict[str, Any]:
    for row in read_csv_rows(path):
        if row["frame_id"] == frame_id:
            return row
    raise SystemExit(f"frame が見つかりません: {frame_id}")


def load_routes(path: Path, page_no: int, frame_id: str) -> list[dict[str, Any]]:
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    return [
        feature for feature in features
        if int(feature["properties"]["page_no"]) == page_no and feature["properties"].get("frame_id") == frame_id
    ]


def pick_pdf_path(explicit: Path | None, routes: list[dict[str, Any]]) -> Path | None:
    if explicit and explicit.exists():
        return explicit
    if routes:
        name = routes[0]["properties"].get("source_pdf")
        if name:
            candidate = Path(name)
            if candidate.exists():
                return candidate
    return None


def render_page_source(
    *,
    pdf_path: Path | None,
    page_no: int,
    page_width_pt: float,
    page_height_pt: float,
    fallback_png: Path,
) -> tuple[Image.Image, str]:
    if pdf_path and pdf_path.exists():
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(FULL_SCALE, FULL_SCALE), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, f"pdf:{pdf_path.name}"
    if fallback_png.exists():
        return Image.open(fallback_png).convert("RGB"), f"fallback_png:{fallback_png.name}"
    width = max(1, int(round(page_width_pt * FULL_SCALE)))
    height = max(1, int(round(page_height_pt * FULL_SCALE)))
    return Image.new("RGB", (width, height), "#faf7f1"), "blank_canvas"


def pdf_to_pixel(point: tuple[float, float], page_width_pt: float, page_height_pt: float, image: Image.Image) -> tuple[float, float]:
    scale_x = image.width / page_width_pt
    scale_y = image.height / page_height_pt
    return (point[0] * scale_x, point[1] * scale_y)


def polyline_midpoint(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    target = total / 2.0
    walked = 0.0
    for (start, end), seg_len in zip(zip(points, points[1:]), lengths):
        if walked + seg_len >= target and seg_len > 0:
            t = (target - walked) / seg_len
            return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        walked += seg_len
    return points[len(points) // 2]


def route_pdf_points(feature: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geom = feature["geometry"]
    if geom["type"] == "LineString":
        return [[synthetic_to_pdf(coord) for coord in geom["coordinates"]]]
    return [[synthetic_to_pdf(coord) for coord in line] for line in geom["coordinates"]]


def draw_route_overlay(
    image: Image.Image,
    routes: list[dict[str, Any]],
    page_width_pt: float,
    page_height_pt: float,
    frame_bbox_pt: tuple[float, float, float, float],
    *,
    crop_origin_px: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    styles = {
        "walk_main": {"color": ImageColor.getrgb("#00a6d6"), "width": 4},
        "walk_sub": {"color": ImageColor.getrgb("#147a2e"), "width": 3},
    }
    x0, y0, x1, y1 = frame_bbox_pt
    frame_pixels = [
        pdf_to_pixel((x0, y0), page_width_pt, page_height_pt, image),
        pdf_to_pixel((x1, y1), page_width_pt, page_height_pt, image),
    ]
    fx0 = frame_pixels[0][0] - crop_origin_px[0]
    fy0 = frame_pixels[0][1] - crop_origin_px[1]
    fx1 = frame_pixels[1][0] - crop_origin_px[0]
    fy1 = frame_pixels[1][1] - crop_origin_px[1]
    draw.rectangle((fx0, fy0, fx1, fy1), outline="#f2c94c", width=3)

    for feature in routes:
        props = feature["properties"]
        style = styles.get(props.get("style_class", ""), {"color": ImageColor.getrgb("#3b82f6"), "width": 3})
        all_pdf_points: list[tuple[float, float]] = []
        for line in route_pdf_points(feature):
            pixels = [
                (
                    pdf_to_pixel(point, page_width_pt, page_height_pt, image)[0] - crop_origin_px[0],
                    pdf_to_pixel(point, page_width_pt, page_height_pt, image)[1] - crop_origin_px[1],
                )
                for point in line
            ]
            all_pdf_points.extend(line)
            if len(pixels) >= 2:
                draw.line(pixels, fill=style["color"] + (230,), width=style["width"], joint="curve")
        mid_pdf = polyline_midpoint(all_pdf_points)
        label_x, label_y = pdf_to_pixel(mid_pdf, page_width_pt, page_height_pt, image)
        label_x -= crop_origin_px[0]
        label_y -= crop_origin_px[1]
        label = props["route_id"]
        bbox = draw.textbbox((label_x, label_y), label, font=font)
        pad = 2
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255, 220), outline=style["color"])
        draw.text((label_x, label_y), label, fill=(20, 20, 20, 255), font=font)
    return out


def red_pixel_ratio(image: Image.Image, pdf_points: list[tuple[float, float]], page_width_pt: float, page_height_pt: float) -> float:
    if not pdf_points:
        return 0.0
    rgb = image.convert("RGB")
    hits = 0
    total = 0
    for point in pdf_points:
        px, py = pdf_to_pixel(point, page_width_pt, page_height_pt, image)
        ix = int(round(px))
        iy = int(round(py))
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                x = ix + ox
                y = iy + oy
                if x < 0 or y < 0 or x >= rgb.width or y >= rgb.height:
                    continue
                total += 1
                r, g, b = rgb.getpixel((x, y))
                if r >= 140 and (r - g) >= 35 and (r - b) >= 35:
                    hits += 1
    return hits / total if total else 0.0


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * ratio) - 1)))
    return float(ordered[idx])


def load_road_eval_rows(base_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summary_by_route: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(base_dir / "route_summary.csv"):
        summary_by_route[row["route_id"]] = row
    samples_by_route: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv_rows(base_dir / "route_samples.csv"):
        samples_by_route.setdefault(row["route_id"], []).append(row)
    return summary_by_route, samples_by_route


def recommended_action(
    extraction_status: str,
    likely_issue: str,
    mean_dist: float,
    p90_dist: float,
    max_dist: float,
    failed_ratio: float,
    snap_ratio: float,
    gcp_pass: bool,
) -> str:
    if extraction_status == "extraction_issue":
        return "extraction_fix_needed"
    if not gcp_pass and max_dist > 80:
        return "georef_gcp_recheck"
    if max_dist < 50 and failed_ratio < 0.05:
        return "accept_as_is"
    if likely_issue == "possible_missing_trail_in_osm":
        return "osm_missing_or_trail"
    if likely_issue == "mixed_alignment_or_missing_path" and snap_ratio < 0.6:
        return "osm_missing_or_trail"
    if p90_dist < 60 and snap_ratio >= 0.7:
        return "map_match_candidate"
    if likely_issue == "likely_georef_misalignment" and gcp_pass:
        return "manual_edit_needed"
    return "manual_edit_needed"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_report(
    path: Path,
    *,
    source_desc: str,
    frame_id: str,
    page_no: int,
    gcp_row: dict[str, Any] | None,
    extraction_summary: dict[str, Any],
    diagnosis_rows: list[dict[str, Any]],
) -> None:
    gcp_pass = bool(gcp_row and gcp_row.get("quality_status") == "pass")
    extraction_ok_count = sum(1 for row in diagnosis_rows if row["pdf_extraction_status"] == "extraction_ok")
    extraction_issue_count = len(diagnosis_rows) - extraction_ok_count
    map_match_ids = [row["route_id"] for row in diagnosis_rows if row["recommended_action"] == "map_match_candidate"]
    manual_ids = [row["route_id"] for row in diagnosis_rows if row["recommended_action"] == "manual_edit_needed"]
    osm_missing_ids = [row["route_id"] for row in diagnosis_rows if row["recommended_action"] == "osm_missing_or_trail"]
    extraction_fix_ids = [row["route_id"] for row in diagnosis_rows if row["recommended_action"] == "extraction_fix_needed"]
    dominant_cause = extraction_summary["dominant_cause"]
    lines = [
        "# PDF Overlay Inspection",
        "",
        f"- Page / Frame: `{page_no}` / `{frame_id}`",
        f"- Source image: `{source_desc}`",
        f"- GCP transform pass: `{'yes' if gcp_pass else 'no'}`",
    ]
    if gcp_row:
        lines.extend(
            [
                f"- Step5 model: `{gcp_row['transform_model']} / {gcp_row['y_mode']} / {gcp_row['crs_candidate']}`",
                f"- gcp_count: `{gcp_row['gcp_count']}`",
                f"- loocv_rmse_m: `{gcp_row['loocv_rmse_m']}`",
            ]
        )
    lines.extend(
        [
            f"- PDF extraction ok routes: `{extraction_ok_count}`",
            f"- PDF extraction issue routes: `{extraction_issue_count}`",
            "",
            "## Final Judgement",
            f"- GCP変換は pass と見てよいか: `{'yes' if gcp_pass else 'no'}`",
            f"- PDF上の抽出は正しいか: `{extraction_summary['overall_status']}`",
            f"- Mapbox道路とズレる主原因: `{dominant_cause}`",
            f"- 次に map matching すべき route_id: `{', '.join(map_match_ids) if map_match_ids else '-'}`",
            f"- 手修正すべき route_id: `{', '.join(manual_ids) if manual_ids else '-'}`",
            f"- OSM未収録っぽい route_id: `{', '.join(osm_missing_ids) if osm_missing_ids else '-'}`",
            f"- 抽出修正が必要な route_id: `{', '.join(extraction_fix_ids) if extraction_fix_ids else '-'}`",
            "",
            "## Outputs",
            "- `pdf_overlay_full.png`",
            "- `pdf_overlay_frame.png`",
            "- `route_diagnosis.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDF/page preview overlay and classify route issues for one frame")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--page-no", type=int, default=61)
    parser.add_argument("--frame-id", type=str, default="061_f06")
    parser.add_argument("--frame-csv", type=Path, default=Path("artifacts/step2/frames.csv"))
    parser.add_argument("--page-meta", type=Path, default=Path("artifacts/step1/page_red_summary.csv"))
    parser.add_argument("--road-eval-dir", type=Path, default=Path("artifacts/manual_georef_test_variants/page_061_061_f06/road_eval"))
    parser.add_argument("--frame-models", type=Path, default=Path("artifacts/step5/frame_models.csv"))
    parser.add_argument("--fallback-page-image", type=Path, default=Path("artifacts/step3/mapbox_debug/pages/page_061.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/inspection/page_061_061_f06"))
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    page_meta = load_page_meta(args.page_meta, args.page_no)
    frame_row = load_frame_row(args.frame_csv, args.frame_id)
    routes = load_routes(args.routes, args.page_no, args.frame_id)
    pdf_path = pick_pdf_path(args.pdf, routes)
    source_image, source_desc = render_page_source(
        pdf_path=pdf_path,
        page_no=args.page_no,
        page_width_pt=float(page_meta["page_width_pt"]),
        page_height_pt=float(page_meta["page_height_pt"]),
        fallback_png=args.fallback_page_image,
    )

    page_width_pt = float(page_meta["page_width_pt"])
    page_height_pt = float(page_meta["page_height_pt"])
    frame_bbox_pt = (
        float(frame_row["x0_pt"]),
        float(frame_row["y0_pt"]),
        float(frame_row["x1_pt"]),
        float(frame_row["y1_pt"]),
    )
    full_overlay = draw_route_overlay(source_image, routes, page_width_pt, page_height_pt, frame_bbox_pt)
    full_overlay.save(args.out_dir / "pdf_overlay_full.png")

    scale_x = source_image.width / page_width_pt
    scale_y = source_image.height / page_height_pt
    crop_box = (
        max(0, int(round((frame_bbox_pt[0] - FRAME_MARGIN_PT) * scale_x))),
        max(0, int(round((frame_bbox_pt[1] - FRAME_MARGIN_PT) * scale_y))),
        min(source_image.width, int(round((frame_bbox_pt[2] + FRAME_MARGIN_PT) * scale_x))),
        min(source_image.height, int(round((frame_bbox_pt[3] + FRAME_MARGIN_PT) * scale_y))),
    )
    crop_origin = (crop_box[0], crop_box[1])
    frame_source = source_image.crop(crop_box)
    frame_overlay = draw_route_overlay(frame_source, routes, page_width_pt, page_height_pt, frame_bbox_pt, crop_origin_px=crop_origin)
    frame_overlay.save(args.out_dir / "pdf_overlay_frame.png")

    summary_by_route, samples_by_route = load_road_eval_rows(args.road_eval_dir)
    frame_models = read_csv_rows(args.frame_models)
    frame_model = next((row for row in frame_models if row["frame_id"] == args.frame_id), None)
    gcp_pass = bool(frame_model and frame_model.get("quality_status") == "pass")

    diagnosis_rows: list[dict[str, Any]] = []
    extraction_issue_count = 0
    for feature in routes:
        route_id = feature["properties"]["route_id"]
        style_class = feature["properties"].get("style_class", "")
        pdf_lines = route_pdf_points(feature)
        flat_points = [point for line in pdf_lines for point in line]
        red_ratio = red_pixel_ratio(source_image, flat_points, page_width_pt, page_height_pt)
        extraction_status = "extraction_ok" if red_ratio >= RED_RATIO_OK else "extraction_issue"
        if extraction_status == "extraction_issue":
            extraction_issue_count += 1

        road_summary = summary_by_route.get(route_id, {})
        sample_rows = samples_by_route.get(route_id, [])
        dists = [float(row["nearest_road_distance_m"]) for row in sample_rows]
        failed_rows = [row for row in sample_rows if row["failed_threshold"] == "True"]
        mean_dist = float(road_summary.get("mean_distance_m", 0.0) or 0.0)
        p90_dist = percentile(dists, 0.9)
        max_dist = float(road_summary.get("max_distance_m", 0.0) or 0.0)
        failed_count = len(failed_rows)
        failed_ratio = (failed_count / len(sample_rows)) if sample_rows else 0.0
        snap_ratio = float(road_summary.get("snap_ratio", 0.0) or 0.0)
        likely_issue = road_summary.get("likely_issue", "")
        action = recommended_action(
            extraction_status,
            likely_issue,
            mean_dist,
            p90_dist,
            max_dist,
            failed_ratio,
            snap_ratio,
            gcp_pass,
        )
        diagnosis_rows.append(
            {
                "route_id": route_id,
                "style_class": style_class,
                "pdf_extraction_status": extraction_status,
                "mean_road_dist_m": round(mean_dist, 3),
                "p90_road_dist_m": round(p90_dist, 3),
                "max_road_dist_m": round(max_dist, 3),
                "failed_sample_count": failed_count,
                "failed_ratio": round(failed_ratio, 4),
                "snap_ratio": round(snap_ratio, 4),
                "likely_issue": likely_issue,
                "recommended_action": action,
            }
        )

    write_csv(args.out_dir / "route_diagnosis.csv", diagnosis_rows)

    cause_counts: dict[str, int] = {}
    for row in diagnosis_rows:
        key = row["recommended_action"]
        cause_counts[key] = cause_counts.get(key, 0) + 1
    dominant_cause = max(cause_counts, key=cause_counts.get) if cause_counts else "unknown"
    extraction_summary = {
        "overall_status": "extraction_issue" if extraction_issue_count > 0 else "extraction_ok",
        "dominant_cause": dominant_cause,
    }
    write_report(
        args.out_dir / "pdf_overlay_report.md",
        source_desc=source_desc,
        frame_id=args.frame_id,
        page_no=args.page_no,
        gcp_row=frame_model,
        extraction_summary=extraction_summary,
        diagnosis_rows=diagnosis_rows,
    )


if __name__ == "__main__":
    main()
