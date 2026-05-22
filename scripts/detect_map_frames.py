#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw


PRIMARY_ROUTE_CLASS = "route_candidate_solid_main"
ROUTE_CLASSES = {PRIMARY_ROUTE_CLASS}
DASHED_CLASSES = {"red_dashed_nonroute", "annotation_dashed"}
ANNOTATION_CLASSES = {"filled_symbol_or_legend", "small_symbol_or_label", "legend_like", "red_annotation_solid", "unknown_red"}
PREVIEW_COLORS = [
    "#00C2FF",
    "#7CFF6B",
    "#FFB703",
    "#FF4D6D",
    "#9B5DE5",
    "#06D6A0",
    "#EF476F",
    "#118AB2",
]


@dataclass
class FrameCandidate:
    page_no: int
    candidate_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    area: float
    draw_type: str
    stroke_rgb: tuple[float, float, float] | None
    fill_rgb: tuple[float, float, float] | None
    stroke_width: float
    has_rect_like: bool
    complexity: int
    route_object_count: int
    dashed_route_count: int
    solid_route_count: int
    red_object_count: int
    filled_symbol_count: int
    unknown_red_count: int
    small_symbol_count: int
    score: float
    source_kind: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def color_tuple(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, (int, float)):
        component = float(value)
        return (component, component, component)
    return None


def mean_color(color: tuple[float, float, float] | None) -> float:
    if color is None:
        return 0.0
    return sum(color) / 3.0


def color_spread(color: tuple[float, float, float] | None) -> float:
    if color is None:
        return 0.0
    return max(color) - min(color)


def is_near_white(color: tuple[float, float, float] | None) -> bool:
    return color is not None and mean_color(color) >= 0.985 and color_spread(color) <= 0.03


def is_pale_map_fill(color: tuple[float, float, float] | None) -> bool:
    if color is None:
        return False
    return min(color) >= 0.75 and mean_color(color) >= 0.82 and not is_near_white(color)


def is_dark_stroke(color: tuple[float, float, float] | None) -> bool:
    return color is not None and mean_color(color) <= 0.28


def is_blueish_stroke(color: tuple[float, float, float] | None) -> bool:
    if color is None:
        return False
    return color[2] >= 0.65 and color[0] <= 0.2 and color[1] <= 0.75


def rect_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def rect_contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float], tolerance: float = 0.0) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    )


def rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = rect_area(a) + rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def rect_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def center_in_rect(cx: float, cy: float, rect: tuple[float, float, float, float]) -> bool:
    return rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def item_counter(items: list[Any]) -> Counter[str]:
    return Counter(item[0] for item in items)


def route_rows_for_page(red_rows: list[dict[str, Any]], page_no: int) -> list[dict[str, Any]]:
    return [row for row in red_rows if row["page_no"] == page_no and row["classification"] in ROUTE_CLASSES]


def all_red_rows_for_page(red_rows: list[dict[str, Any]], page_no: int) -> list[dict[str, Any]]:
    return [row for row in red_rows if row["page_no"] == page_no]


def route_center(row: dict[str, Any]) -> tuple[float, float]:
    return ((row["bbox_x0_pt"] + row["bbox_x1_pt"]) / 2.0, (row["bbox_y0_pt"] + row["bbox_y1_pt"]) / 2.0)


def route_bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (row["bbox_x0_pt"], row["bbox_y0_pt"], row["bbox_x1_pt"], row["bbox_y1_pt"])


def gather_counts_in_rect(rows: list[dict[str, Any]], rect: tuple[float, float, float, float]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        cx, cy = route_center(row)
        if center_in_rect(cx, cy, rect):
            counts[row["classification"]] += 1
    return counts


def compute_candidate_score(
    rect: fitz.Rect,
    draw_type: str,
    stroke_rgb: tuple[float, float, float] | None,
    fill_rgb: tuple[float, float, float] | None,
    stroke_width: float,
    counts: Counter[str],
    has_rect_like: bool,
    complexity: int,
    page_rect: fitz.Rect,
) -> float:
    route_count = counts[PRIMARY_ROUTE_CLASS]
    route_density = route_count / max(rect.width * rect.height, 1.0)
    score = 0.0
    if is_pale_map_fill(fill_rgb):
        score += 3.0
    if is_dark_stroke(stroke_rgb) and stroke_width >= 0.3:
        score += 2.5 if stroke_width >= 1.2 else 1.5
    if is_blueish_stroke(stroke_rgb) and stroke_width >= 1.0:
        score += 0.8
    if has_rect_like:
        score += 0.6
    if draw_type in {"f", "fs"} and complexity >= 6:
        score += 0.6
    score += min(route_count, 40) * 0.18
    score += min(route_density * 120000.0, 1.6)
    if is_near_white(fill_rgb):
        score -= 1.5
    if rect.width > page_rect.width * 0.95 and rect.height > page_rect.height * 0.4:
        score -= 2.0
    aspect = rect.width / max(rect.height, 1.0)
    if aspect > 5.5 or aspect < 0.18:
        score -= 1.5
    if complexity <= 2 and not has_rect_like:
        score -= 2.5
    if route_count == 0:
        score -= 3.0
    return score


def extract_frame_candidates(page: fitz.Page, red_rows: list[dict[str, Any]]) -> list[FrameCandidate]:
    page_no = page.number + 1
    page_rect = page.rect
    all_page_red = all_red_rows_for_page(red_rows, page_no)
    candidates: list[FrameCandidate] = []

    for draw_index, drawing in enumerate(page.get_drawings()):
        rect = drawing.get("rect")
        if rect is None:
            continue
        if rect.width < 90 or rect.height < 60:
            continue
        if rect.x1 < 0 or rect.y1 < 0 or rect.x0 > page_rect.width or rect.y0 > page_rect.height:
            continue

        items = drawing.get("items", [])
        ops = item_counter(items)
        complexity = len(items)
        has_rect_like = "re" in ops or "qu" in ops
        if not has_rect_like and complexity < 4:
            continue

        stroke_rgb = color_tuple(drawing.get("color"))
        fill_rgb = color_tuple(drawing.get("fill"))
        counts = gather_counts_in_rect(all_page_red, (rect.x0, rect.y0, rect.x1, rect.y1))
        route_count = counts[PRIMARY_ROUTE_CLASS]
        route_density = route_count / max(rect.width * rect.height, 1.0)
        score = compute_candidate_score(
            rect=rect,
            draw_type=drawing.get("type"),
            stroke_rgb=stroke_rgb,
            fill_rgb=fill_rgb,
            stroke_width=float(drawing.get("width") or 0.0),
            counts=counts,
            has_rect_like=has_rect_like,
            complexity=complexity,
            page_rect=page_rect,
        )
        if route_count < 3:
            continue
        if score < 3.2:
            continue

        source_kind = "vector_bbox"
        if is_pale_map_fill(fill_rgb):
            source_kind = "filled_region"
        elif is_dark_stroke(stroke_rgb):
            source_kind = "dark_border"

        if source_kind == "vector_bbox" and not has_rect_like:
            continue
        if (
            source_kind == "filled_region"
            and not has_rect_like
            and rect.width > page_rect.width * 0.55
            and route_density < 0.00022
        ):
            continue

        candidates.append(
            FrameCandidate(
                page_no=page_no,
                candidate_id=draw_index,
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
                width=float(rect.width),
                height=float(rect.height),
                area=float(rect.width * rect.height),
                draw_type=drawing.get("type"),
                stroke_rgb=stroke_rgb,
                fill_rgb=fill_rgb,
                stroke_width=float(drawing.get("width") or 0.0),
                has_rect_like=has_rect_like,
                complexity=complexity,
                route_object_count=route_count,
                dashed_route_count=sum(counts[name] for name in DASHED_CLASSES),
                solid_route_count=counts[PRIMARY_ROUTE_CLASS],
                red_object_count=sum(counts.values()),
                filled_symbol_count=counts["filled_symbol_or_legend"],
                unknown_red_count=counts["unknown_red"],
                small_symbol_count=counts["small_symbol_or_label"],
                score=score,
                source_kind=source_kind,
            )
        )

    return candidates


def dedupe_candidates(candidates: list[FrameCandidate]) -> list[FrameCandidate]:
    selected: list[FrameCandidate] = []
    ordered = sorted(candidates, key=lambda candidate: (candidate.score, candidate.route_object_count, -candidate.area), reverse=True)
    for candidate in ordered:
        matched = False
        for existing in selected:
            if rect_iou(candidate.bbox, existing.bbox) >= 0.88:
                matched = True
                break
            if (
                abs(candidate.x0 - existing.x0) <= 8
                and abs(candidate.y0 - existing.y0) <= 8
                and abs(candidate.x1 - existing.x1) <= 8
                and abs(candidate.y1 - existing.y1) <= 8
            ):
                matched = True
                break
        if not matched:
            selected.append(candidate)
    return selected


def build_route_supports(route_rows: list[dict[str, Any]], candidates: list[FrameCandidate]) -> dict[tuple[int, int], set[int]]:
    supports: dict[tuple[int, int], set[int]] = {}
    for candidate in candidates:
        key = (candidate.page_no, candidate.candidate_id)
        support: set[int] = set()
        for index, row in enumerate(route_rows):
            cx, cy = route_center(row)
            if center_in_rect(cx, cy, candidate.bbox):
                support.add(index)
        supports[key] = support
    return supports


def suppress_parent_frames(
    candidates: list[FrameCandidate],
    supports: dict[tuple[int, int], set[int]],
) -> list[FrameCandidate]:
    suppressed_ids: set[tuple[int, int]] = set()
    for parent in candidates:
        parent_key = (parent.page_no, parent.candidate_id)
        parent_support = supports.get(parent_key, set())
        if len(parent_support) < 6:
            continue
        children = [
            child for child in candidates
            if child != parent
            and child.area < parent.area * 0.92
            and rect_contains(parent.bbox, child.bbox, tolerance=6.0)
        ]
        if len(children) < 2:
            continue
        strong_children = []
        union_support: set[int] = set()
        for child in children:
            child_key = (child.page_no, child.candidate_id)
            child_support = supports.get(child_key, set())
            if len(child_support) < 3:
                continue
            if len(child_support) < max(3, int(len(parent_support) * 0.22)):
                continue
            strong_children.append(child)
            union_support |= child_support
        if len(strong_children) < 2:
            continue
        if len(union_support) >= int(len(parent_support) * 0.78):
            suppressed_ids.add(parent_key)

    return [candidate for candidate in candidates if (candidate.page_no, candidate.candidate_id) not in suppressed_ids]


def suppress_inner_legend_candidates(
    candidates: list[FrameCandidate],
    supports: dict[tuple[int, int], set[int]],
) -> list[FrameCandidate]:
    suppressed_ids: set[tuple[int, int]] = set()
    for child in candidates:
        child_key = (child.page_no, child.candidate_id)
        child_support = supports.get(child_key, set())
        for parent in candidates:
            if parent == child:
                continue
            parent_key = (parent.page_no, parent.candidate_id)
            parent_support = supports.get(parent_key, set())
            if not rect_contains(parent.bbox, child.bbox, tolerance=8.0):
                continue
            if parent.area <= child.area * 1.4:
                continue
            if len(child_support) == 0 or len(parent_support) == 0:
                continue
            if len(child_support - parent_support) > 0:
                continue
            if not is_pale_map_fill(parent.fill_rgb) and not is_dark_stroke(parent.stroke_rgb):
                continue
            if not (is_near_white(child.fill_rgb) or (is_blueish_stroke(child.stroke_rgb) and not is_dark_stroke(child.stroke_rgb))):
                continue
            if len(parent_support) >= max(6, int(len(child_support) * 1.3)):
                suppressed_ids.add(child_key)
                break
    return [candidate for candidate in candidates if (candidate.page_no, candidate.candidate_id) not in suppressed_ids]


def cluster_route_rows(route_rows: list[dict[str, Any]], gap_pt: float = 52.0) -> list[list[dict[str, Any]]]:
    if not route_rows:
        return []
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left_index, left_row in enumerate(route_rows):
        left_bbox = route_bbox(left_row)
        for right_index in range(left_index + 1, len(route_rows)):
            right_bbox = route_bbox(route_rows[right_index])
            if rect_distance(left_bbox, right_bbox) <= gap_pt:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    clusters: list[list[dict[str, Any]]] = []
    visited: set[int] = set()
    for index in range(len(route_rows)):
        if index in visited:
            continue
        stack = [index]
        component: list[dict[str, Any]] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(route_rows[current])
            stack.extend(adjacency[current] - visited)
        clusters.append(component)
    return clusters


def cluster_bbox(cluster: list[dict[str, Any]], margin: float = 18.0) -> tuple[float, float, float, float]:
    x0 = min(row["bbox_x0_pt"] for row in cluster) - margin
    y0 = min(row["bbox_y0_pt"] for row in cluster) - margin
    x1 = max(row["bbox_x1_pt"] for row in cluster) + margin
    y1 = max(row["bbox_y1_pt"] for row in cluster) + margin
    return (x0, y0, x1, y1)


def snap_cluster_to_candidate(
    cluster_box: tuple[float, float, float, float],
    candidates: list[FrameCandidate],
) -> FrameCandidate | None:
    cluster_area = rect_area(cluster_box)
    cluster_center_x = (cluster_box[0] + cluster_box[2]) / 2.0
    cluster_center_y = (cluster_box[1] + cluster_box[3]) / 2.0
    matching: list[tuple[float, FrameCandidate]] = []

    for candidate in candidates:
        if not rect_contains(candidate.bbox, cluster_box, tolerance=12.0):
            continue
        if not center_in_rect(cluster_center_x, cluster_center_y, candidate.bbox):
            continue
        area_ratio = candidate.area / max(cluster_area, 1.0)
        if area_ratio > 28.0:
            continue
        bonus = candidate.score
        if is_pale_map_fill(candidate.fill_rgb):
            bonus += 1.0
        if is_dark_stroke(candidate.stroke_rgb):
            bonus += 0.8
        bonus -= area_ratio * 0.08
        matching.append((bonus, candidate))

    if not matching:
        return None
    matching.sort(key=lambda item: (item[0], -item[1].area), reverse=True)
    return matching[0][1]


def fallback_frame_from_cluster(page_no: int, cluster_id: int, cluster: list[dict[str, Any]], page_rect: fitz.Rect) -> FrameCandidate:
    x0, y0, x1, y1 = cluster_bbox(cluster, margin=22.0)
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(page_rect.width), x1)
    y1 = min(float(page_rect.height), y1)
    dashed_count = sum(1 for row in cluster if row["classification"] in DASHED_CLASSES)
    solid_count = sum(1 for row in cluster if row["classification"] == PRIMARY_ROUTE_CLASS)
    return FrameCandidate(
        page_no=page_no,
        candidate_id=100000 + cluster_id,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width=x1 - x0,
        height=y1 - y0,
        area=(x1 - x0) * (y1 - y0),
        draw_type="cluster",
        stroke_rgb=None,
        fill_rgb=None,
        stroke_width=0.0,
        has_rect_like=False,
        complexity=len(cluster),
        route_object_count=len(cluster),
        dashed_route_count=dashed_count,
        solid_route_count=solid_count,
        red_object_count=len(cluster),
        filled_symbol_count=0,
        unknown_red_count=0,
        small_symbol_count=0,
        score=2.5 + len(cluster) * 0.2,
        source_kind="cluster_fallback",
    )


def merge_selected_frames(frames: list[FrameCandidate]) -> list[FrameCandidate]:
    merged: list[FrameCandidate] = []
    for frame in sorted(frames, key=lambda item: (item.page_no, item.y0, item.x0, -item.area)):
        duplicate = False
        for existing in merged:
            if frame.page_no != existing.page_no:
                continue
            if rect_iou(frame.bbox, existing.bbox) >= 0.84:
                duplicate = True
                break
            if rect_contains(existing.bbox, frame.bbox, tolerance=10.0) and frame.route_object_count <= existing.route_object_count:
                duplicate = True
                break
        if not duplicate:
            merged.append(frame)
    return merged


def greedy_select_frames(
    candidates: list[FrameCandidate],
    supports: dict[tuple[int, int], set[int]],
) -> list[FrameCandidate]:
    uncovered: set[int] = set()
    for support in supports.values():
        uncovered |= support

    selected: list[FrameCandidate] = []
    remaining = list(candidates)
    while remaining and uncovered:
        scored: list[tuple[float, int, FrameCandidate]] = []
        for candidate in remaining:
            support = supports.get((candidate.page_no, candidate.candidate_id), set())
            new_support = uncovered & support
            if len(new_support) == 0:
                continue
            area_penalty = candidate.area / 70000.0
            value = len(new_support) * 2.6 + candidate.score - area_penalty
            scored.append((value, len(new_support), candidate))
        if not scored:
            break
        scored.sort(key=lambda item: (item[0], item[1], -item[2].area), reverse=True)
        best_value, best_new_support, best_candidate = scored[0]
        if best_new_support < 2 and best_candidate.route_object_count < 5:
            break
        selected.append(best_candidate)
        uncovered -= supports.get((best_candidate.page_no, best_candidate.candidate_id), set())
        remaining = [
            candidate for candidate in remaining
            if candidate != best_candidate
        ]
    return selected


def suppress_redundant_selected_frames(
    selected: list[FrameCandidate],
    supports: dict[tuple[int, int], set[int]],
) -> list[FrameCandidate]:
    suppressed_ids: set[tuple[int, int]] = set()
    for parent in selected:
        parent_key = (parent.page_no, parent.candidate_id)
        parent_support = supports.get(parent_key, set())
        if len(parent_support) < 6:
            continue
        union_support: set[int] = set()
        contributors = 0
        for child in selected:
            if child == parent:
                continue
            if child.area >= parent.area * 0.88:
                continue
            child_key = (child.page_no, child.candidate_id)
            child_support = supports.get(child_key, set())
            overlap = child_support & parent_support
            if len(overlap) < max(3, int(len(parent_support) * 0.18)):
                continue
            contributors += 1
            union_support |= overlap
        if contributors >= 2 and len(union_support) >= int(len(parent_support) * 0.68):
            suppressed_ids.add(parent_key)
    return [frame for frame in selected if (frame.page_no, frame.candidate_id) not in suppressed_ids]


def assign_frame_id(row: dict[str, Any], frames: list[FrameCandidate]) -> str | None:
    cx, cy = route_center(row)
    row_box = route_bbox(row)
    containing = [
        frame for frame in frames
        if center_in_rect(cx, cy, frame.bbox)
    ]
    if containing:
        containing.sort(key=lambda frame: (frame.area, -frame.score))
        frame = containing[0]
        return f"{frame.page_no:03d}_{frame.candidate_id}"

    nearest: list[tuple[float, FrameCandidate]] = []
    for frame in frames:
        distance = rect_distance(row_box, frame.bbox)
        if distance <= 24.0:
            nearest.append((distance, frame))
    if nearest:
        nearest.sort(key=lambda item: (item[0], item[1].area))
        return f"{nearest[0][1].page_no:03d}_{nearest[0][1].candidate_id}"
    return None


def load_red_rows(step1_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (step1_dir / "red_objects.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in [
                "page_no",
                "draw_index",
                "width_pt",
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
            ]:
                if field in {"page_no", "draw_index", "segment_count", "curve_count", "closed_count"}:
                    row[field] = int(float(row[field]))
                else:
                    row[field] = float(row[field])
            rows.append(row)
    return rows


def overlay_frame(draw: ImageDraw.ImageDraw, frame: FrameCandidate, scale: float, label: str, color: str) -> None:
    rect = [frame.x0 * scale, frame.y0 * scale, frame.x1 * scale, frame.y1 * scale]
    draw.rectangle(rect, outline=color, width=5)
    tag_box = [rect[0], rect[1] - 22, rect[0] + 82, rect[1]]
    draw.rectangle(tag_box, fill=color)
    draw.text((rect[0] + 6, rect[1] - 20), label, fill="white")


def overlay_route_object(draw: ImageDraw.ImageDraw, row: dict[str, Any], scale: float, color: str) -> None:
    rect = [
        row["bbox_x0_pt"] * scale,
        row["bbox_y0_pt"] * scale,
        row["bbox_x1_pt"] * scale,
        row["bbox_y1_pt"] * scale,
    ]
    width = 3 if row["classification"] in DASHED_CLASSES else 2
    draw.rectangle(rect, outline=color, width=width)


def build_step2(
    pdf_path: Path,
    step1_dir: Path,
    out_dir: Path,
    preview_pages: int,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    previews_dir = out_dir / "previews"
    ensure_dir(previews_dir)

    red_rows = load_red_rows(step1_dir)
    doc = fitz.open(pdf_path)

    all_frames: list[FrameCandidate] = []
    page_summary_rows: list[dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_no = page_index + 1
        route_rows = route_rows_for_page(red_rows, page_no)
        deduped_candidates = dedupe_candidates(extract_frame_candidates(page, red_rows))
        supports = build_route_supports(route_rows, deduped_candidates)
        vector_candidates = suppress_inner_legend_candidates(
            suppress_parent_frames(deduped_candidates, supports),
            supports,
        )
        candidate_supports = build_route_supports(route_rows, vector_candidates)
        selected_frames = suppress_redundant_selected_frames(
            greedy_select_frames(vector_candidates, candidate_supports),
            candidate_supports,
        )

        covered_route_indexes: set[int] = set()
        for frame in selected_frames:
            covered_route_indexes |= candidate_supports.get((frame.page_no, frame.candidate_id), set())
        uncovered_route_rows = [row for idx, row in enumerate(route_rows) if idx not in covered_route_indexes]
        clusters = cluster_route_rows(uncovered_route_rows, gap_pt=34.0)
        for cluster_index, cluster in enumerate(clusters, start=1):
            if len(cluster) == 0:
                continue
            if len(cluster) < 2:
                continue
            selected_frames.append(fallback_frame_from_cluster(page_no, cluster_index, cluster, page.rect))

        merged_frames = merge_selected_frames(selected_frames)
        all_frames.extend(merged_frames)

        page_summary_rows.append(
            {
                "page_no": page_no,
                "route_cluster_count": len(clusters),
                "vector_candidate_count": len(vector_candidates),
                "selected_frame_count": len(merged_frames),
                "route_object_count": len(route_rows),
                "frame_ids": json.dumps([f"{frame.page_no:03d}_{frame.candidate_id}" for frame in merged_frames], ensure_ascii=False),
            }
        )

    frames_by_page: dict[int, list[FrameCandidate]] = defaultdict(list)
    for frame in all_frames:
        frames_by_page[frame.page_no].append(frame)
    for page_frames in frames_by_page.values():
        page_frames.sort(key=lambda frame: (frame.y0, frame.x0, -frame.area))

    frame_id_map: dict[tuple[int, int], str] = {}
    frames_csv_rows: list[dict[str, Any]] = []
    for page_no, page_frames in sorted(frames_by_page.items()):
        for index, frame in enumerate(page_frames, start=1):
            frame_id = f"{page_no:03d}_f{index:02d}"
            frame_id_map[(page_no, frame.candidate_id)] = frame_id
            frames_csv_rows.append(
                {
                    "frame_id": frame_id,
                    "page_no": page_no,
                    "source_kind": frame.source_kind,
                    "draw_type": frame.draw_type,
                    "x0_pt": round(frame.x0, 3),
                    "y0_pt": round(frame.y0, 3),
                    "x1_pt": round(frame.x1, 3),
                    "y1_pt": round(frame.y1, 3),
                    "width_pt": round(frame.width, 3),
                    "height_pt": round(frame.height, 3),
                    "area_pt2": round(frame.area, 3),
                    "route_object_count": frame.route_object_count,
                    "dashed_route_count": frame.dashed_route_count,
                    "solid_route_count": frame.solid_route_count,
                    "red_object_count": frame.red_object_count,
                    "filled_symbol_count": frame.filled_symbol_count,
                    "unknown_red_count": frame.unknown_red_count,
                    "small_symbol_count": frame.small_symbol_count,
                    "score": round(frame.score, 3),
                }
            )

    enriched_red_rows: list[dict[str, Any]] = []
    for row in red_rows:
        page_frames = frames_by_page.get(row["page_no"], [])
        assigned = assign_frame_id(row, page_frames)
        if assigned is not None:
            page_no = int(assigned.split("_")[0])
            candidate_id = int(assigned.split("_")[1])
            row_frame_id = frame_id_map.get((page_no, candidate_id))
        else:
            row_frame_id = None
        enriched_red_rows.append(row | {"frame_id": row_frame_id})

    preview_target_pages = [
        row["page_no"]
        for row in sorted(
            page_summary_rows,
            key=lambda row: (row["selected_frame_count"], row["route_object_count"], row["vector_candidate_count"]),
            reverse=True,
        )[:preview_pages]
    ]

    preview_rows_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_red_rows:
        preview_rows_by_page[row["page_no"]].append(row)

    preview_log: list[dict[str, Any]] = []
    for page_no in preview_target_pages:
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image, "RGBA")
        scale = 2.0
        page_frames = frames_by_page.get(page_no, [])
        frame_color_map: dict[str, str] = {}
        for index, frame in enumerate(page_frames):
            frame_id = frame_id_map[(page_no, frame.candidate_id)]
            color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
            frame_color_map[frame_id] = color
            overlay_frame(draw, frame, scale, frame_id, color)

        for row in preview_rows_by_page.get(page_no, []):
            frame_id = row.get("frame_id")
            if row["classification"] == PRIMARY_ROUTE_CLASS and frame_id:
                overlay_route_object(draw, row, scale, "#DC2626")
            elif row["classification"] in DASHED_CLASSES:
                overlay_route_object(draw, row, scale, "#2563EB")
            elif row["classification"] in ANNOTATION_CLASSES:
                overlay_route_object(draw, row, scale, "#6B7280")

        preview_path = previews_dir / f"page_{page_no:03d}.png"
        image.save(preview_path)
        preview_log.append(
            {
                "page_no": page_no,
                "preview_path": str(preview_path),
                "selected_frame_count": len(page_frames),
            }
        )

    frames_csv = out_dir / "frames.csv"
    red_with_frames_csv = out_dir / "red_objects_with_frames.csv"
    page_summary_csv = out_dir / "page_frame_summary.csv"
    report_md = out_dir / "step2_report.md"
    log_json = out_dir / "extraction_log.json"

    with frames_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(frames_csv_rows[0].keys()) if frames_csv_rows else [
            "frame_id",
            "page_no",
            "source_kind",
            "draw_type",
            "x0_pt",
            "y0_pt",
            "x1_pt",
            "y1_pt",
            "width_pt",
            "height_pt",
            "area_pt2",
            "route_object_count",
            "dashed_route_count",
            "solid_route_count",
            "red_object_count",
            "filled_symbol_count",
            "unknown_red_count",
            "small_symbol_count",
            "score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frames_csv_rows)

    with red_with_frames_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(enriched_red_rows[0].keys()) if enriched_red_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_red_rows)

    with page_summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(page_summary_rows[0].keys()) if page_summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(page_summary_rows)

    report_lines = [
        "# Step 2 Report",
        "",
        f"- PDF: `{pdf_path.name}`",
        f"- Pages scanned: `{doc.page_count}`",
        f"- Frames detected: `{len(frames_csv_rows)}`",
        f"- Pages with at least one frame: `{sum(1 for row in page_summary_rows if row['selected_frame_count'] > 0)}`",
        "",
        "## Representative Pages",
    ]
    for preview in preview_log:
        report_lines.append(
            f"- Page {preview['page_no']}: `{Path(preview['preview_path']).name}` ({preview['selected_frame_count']} frames)"
        )
    report_lines.extend(
        [
            "",
            "## Heuristics",
            "- Pale map fills and dark map borders are preferred over blue bordered inner boxes.",
            "- Parent containers are dropped when they mostly contain multiple child map candidates.",
            "- Remaining route clusters without a reliable vector frame fall back to an expanded cluster bbox.",
        ]
    )
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result = {
        "pdf_path": str(pdf_path),
        "page_count": doc.page_count,
        "frames_detected": len(frames_csv_rows),
        "pages_with_frames": sum(1 for row in page_summary_rows if row["selected_frame_count"] > 0),
        "preview_pages": preview_target_pages,
        "outputs": {
            "frames_csv": str(frames_csv),
            "red_objects_with_frames_csv": str(red_with_frames_csv),
            "page_summary_csv": str(page_summary_csv),
            "report_md": str(report_md),
            "previews_dir": str(previews_dir),
        },
    }
    with log_json.open("w", encoding="utf-8") as handle:
        json.dump(result | {"previews": preview_log}, handle, ensure_ascii=False, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect map frames and assign frame_id to Step 1 red objects.")
    parser.add_argument("pdf", type=Path, help="Path to the target PDF")
    parser.add_argument("--step1-dir", type=Path, default=Path("artifacts/step1"), help="Directory containing Step 1 outputs")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/step2"), help="Output directory")
    parser.add_argument("--preview-pages", type=int, default=6, help="Number of preview pages")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_step2(
        pdf_path=args.pdf,
        step1_dir=args.step1_dir,
        out_dir=args.out_dir,
        preview_pages=args.preview_pages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
