#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont


TARGET_BLACK = 2301728
TARGET_GRAY = 11804046
TARGET_WHITE = 16777215
TEMPLE_COLORS = {15539236, 16089631, 29372}
TARGET_DASHES = {"[ .05 2.5 ] 0", "[ .02 3 ] 0"}
KANJI_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
NORMALIZE_TABLE = str.maketrans(
    {
        "圓": "円",
        "寶": "宝",
        "德": "徳",
        "龜": "亀",
        "瀧": "滝",
        "樂": "楽",
        "龍": "竜",
        "會": "会",
        "驛": "駅",
        "舊": "旧",
        "靈": "霊",
        "ヶ": "ケ",
        "ヵ": "カ",
        "之": "の",
        "海": "海",
        "神": "神",
        "﨑": "崎",
    }
)

NUMBER_RE = re.compile(r"四国霊場第\s*([0-9０-９]+)\s*番(奥の院|奥之院)?")
BEKKAKU_RE = re.compile(r"四国別格霊場第\s*([0-9０-９]+)\s*番(奥の院|奥之院)?")
HIRAGANA_RE = re.compile(r"^[ぁ-んー・]+$")
TEMPLE_NAME_HINT_RE = re.compile(r"[寺院社堂庵坊観音神宮権現明王]")


@dataclass
class Frame:
    frame_id: str
    page_no: int
    source_kind: str
    bbox: tuple[float, float, float, float]
    area: float


@dataclass
class GazetteerEntry:
    temple_group: str
    temple_no: int
    name_full: str
    name_short: str
    reading: str
    latitude: float
    longitude: float
    aliases: set[str]
    source: str


@dataclass
class LineRecord:
    label_id: str
    page_no: int
    frame_id: str | None
    text: str
    bbox: tuple[float, float, float, float]
    color: int
    font_size: float
    role: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class DrawCandidate:
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    draw_type: str
    width_pt: float
    stroke_rgb: tuple[float, float, float] | None
    fill_rgb: tuple[float, float, float] | None
    item_count: int
    route_like: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def bbox_contains_point(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return inter / area


def rgb_int_to_tuple(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def rgb_float_to_hex(value: tuple[float, float, float] | None) -> str:
    if not value:
        return ""
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, round(value[0] * 255))),
        max(0, min(255, round(value[1] * 255))),
        max(0, min(255, round(value[2] * 255))),
    )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(NORMALIZE_TABLE)
    normalized = normalized.replace(" ", "").replace("　", "")
    normalized = normalized.replace("（", "").replace("）", "")
    normalized = normalized.replace("ヶ", "ケ")
    return normalized


def number_from_text(text: str) -> int | None:
    digits = normalize_text(text).translate(KANJI_DIGITS)
    match = re.search(r"第([0-9]+)番", digits)
    if not match:
        return None
    return int(match.group(1))


def short_name(full_name: str) -> str:
    cleaned = re.sub(r"（[^）]+）", "", full_name.strip())
    parts = re.split(r"[ 　]+", cleaned)
    return parts[-1] if parts else full_name.strip()


def alias_set(*texts: str) -> set[str]:
    aliases: set[str] = set()
    for text in texts:
        if not text:
            continue
        base = normalize_text(text)
        aliases.add(base)
        aliases.add(base.replace("ケ", "ヶ"))
        aliases.add(base.replace("ヶ", "ケ"))
        aliases.add(re.sub(r"\([^)]*\)", "", base))
        aliases.add(re.sub(r"（[^）]*）", "", base))
    return aliases


def load_frames(path: Path) -> dict[int, list[Frame]]:
    frames_by_page: dict[int, list[Frame]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = Frame(
                frame_id=row["frame_id"],
                page_no=int(row["page_no"]),
                source_kind=row["source_kind"],
                bbox=(
                    float(row["x0_pt"]),
                    float(row["y0_pt"]),
                    float(row["x1_pt"]),
                    float(row["y1_pt"]),
                ),
                area=float(row["area_pt2"]),
            )
            frames_by_page[frame.page_no].append(frame)
    for page_frames in frames_by_page.values():
        page_frames.sort(key=lambda item: item.area)
    return frames_by_page


def assign_frame_id(
    frames: list[Frame],
    bbox: tuple[float, float, float, float],
) -> str | None:
    center = bbox_center(bbox)
    containing = [frame for frame in frames if bbox_contains_point(frame.bbox, center)]
    if containing:
        return containing[0].frame_id
    best_overlap = 0.0
    best_frame_id: str | None = None
    for frame in frames:
        overlap = bbox_overlap_ratio(bbox, frame.bbox)
        if overlap > best_overlap:
            best_overlap = overlap
            best_frame_id = frame.frame_id
    return best_frame_id if best_overlap >= 0.25 else None


def read_mapbox_token() -> str:
    if os.environ.get("MAPBOX_ACCESS_TOKEN"):
        return os.environ["MAPBOX_ACCESS_TOKEN"]
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MAPBOX_ACCESS_TOKEN="):
                return line.partition("=")[2].strip()
    return ""


def build_88_gazetteer(path: Path) -> list[GazetteerEntry]:
    rows: list[GazetteerEntry] = []
    with path.open("r", encoding="cp932", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            full = row["名称"].strip()
            short = short_name(full)
            reading = short_name(row["読み"].strip())
            rows.append(
                GazetteerEntry(
                    temple_group="88",
                    temple_no=index,
                    name_full=full,
                    name_short=short,
                    reading=reading,
                    latitude=float(row["緯度"]),
                    longitude=float(row["経度"]),
                    aliases=alias_set(full, short, reading),
                    source="data_sources/step4/88.csv",
                )
            )
    return rows


def build_bekkaku_gazetteer(path: Path) -> list[GazetteerEntry]:
    html_text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"\{lat:(?P<lat>[-0-9.]+), lng:(?P<lng>[-0-9.]+), content:'<h3>(?P<name>[^<]+)</h3><p>(?P<num>[0-9]+)番札所"
    )
    rows: list[GazetteerEntry] = []
    for match in pattern.finditer(html_text):
        full = html.unescape(match.group("name")).strip()
        short = short_name(full)
        rows.append(
            GazetteerEntry(
                temple_group="bekkaku",
                temple_no=int(match.group("num")),
                name_full=full,
                name_short=short,
                reading="",
                latitude=float(match.group("lat")),
                longitude=float(match.group("lng")),
                aliases=alias_set(full, short),
                source="data_sources/step4/bekkaku20_portal.html",
            )
        )
    return rows


def line_role(text: str, color: int) -> str:
    if NUMBER_RE.search(text):
        return "temple_number"
    if BEKKAKU_RE.search(text):
        return "bekkaku_number"
    stripped = text.strip()
    if color in TEMPLE_COLORS and HIRAGANA_RE.fullmatch(stripped):
        return "temple_reading"
    if color in TEMPLE_COLORS and TEMPLE_NAME_HINT_RE.search(stripped):
        return "temple_name"
    if color not in {TARGET_BLACK, TARGET_GRAY, TARGET_WHITE}:
        return "colored_label"
    return "other"


def extract_lines(
    doc: fitz.Document,
    frames_by_page: dict[int, list[Frame]],
) -> tuple[list[LineRecord], dict[int, list[LineRecord]]]:
    records: list[LineRecord] = []
    page_map: dict[int, list[LineRecord]] = defaultdict(list)
    label_seq = 1
    for page_index in range(doc.page_count):
        page_no = page_index + 1
        page = doc[page_index]
        frames = frames_by_page.get(page_no, [])
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span["text"].strip()]
                if not spans:
                    continue
                text = "".join(span["text"] for span in spans).strip()
                boxes = [tuple(float(v) for v in span["bbox"]) for span in spans]
                bbox = bbox_union(boxes)
                colors = Counter(span["color"] for span in spans)
                font_size = max(float(span["size"]) for span in spans)
                color = colors.most_common(1)[0][0]
                frame_id = assign_frame_id(frames, bbox)
                record = LineRecord(
                    label_id=f"lbl_{label_seq:05d}",
                    page_no=page_no,
                    frame_id=frame_id,
                    text=text,
                    bbox=bbox,
                    color=color,
                    font_size=font_size,
                    role=line_role(text, color),
                )
                records.append(record)
                page_map[page_no].append(record)
                label_seq += 1
    return records, page_map


def build_draw_candidates(page: fitz.Page) -> list[DrawCandidate]:
    candidates: list[DrawCandidate] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        width = float(drawing.get("width") or 0.0)
        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        if box_w <= 0.3 or box_h <= 0.3:
            continue
        if box_w > 28 or box_h > 28:
            continue
        dashes = str(drawing.get("dashes") or "").strip()
        route_like = width >= 1.0 and dashes in TARGET_DASHES
        stroke = drawing.get("color")
        fill = drawing.get("fill")
        candidates.append(
            DrawCandidate(
                center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                bbox=bbox,
                draw_type=str(drawing.get("type") or ""),
                width_pt=width,
                stroke_rgb=tuple(float(x) for x in stroke) if stroke else None,
                fill_rgb=tuple(float(x) for x in fill) if fill else None,
                item_count=len(drawing.get("items") or []),
                route_like=route_like,
            )
        )
    return candidates


def rgb_brightness(rgb: tuple[float, float, float] | None) -> float:
    if rgb is None:
        return 1.0
    return (rgb[0] + rgb[1] + rgb[2]) / 3.0


def rgb_redness(rgb: tuple[float, float, float] | None) -> float:
    if rgb is None:
        return 0.0
    return rgb[0] - ((rgb[1] + rgb[2]) / 2.0)


def snap_to_marker(
    label_center: tuple[float, float],
    frame_id: str | None,
    draws: list[DrawCandidate],
) -> dict[str, Any]:
    scored: list[tuple[float, DrawCandidate]] = []
    for draw in draws:
        dist = math.dist(label_center, draw.center)
        if dist > 42.0:
            continue
        bbox_w = draw.bbox[2] - draw.bbox[0]
        bbox_h = draw.bbox[3] - draw.bbox[1]
        score = max(0.0, 40.0 - dist) / 10.0
        if draw.draw_type in {"f", "fs"}:
            score += 1.1
        if draw.draw_type == "s":
            score += 0.2
        if 1.5 <= bbox_w <= 12.0 and 1.5 <= bbox_h <= 12.0:
            score += 1.0
        elif bbox_w <= 18.0 and bbox_h <= 18.0:
            score += 0.4
        darkness = min(rgb_brightness(draw.fill_rgb), rgb_brightness(draw.stroke_rgb))
        if darkness < 0.45:
            score += 1.1
        if rgb_redness(draw.fill_rgb) > 0.25 or rgb_redness(draw.stroke_rgb) > 0.25:
            score += 0.5
        if 2 <= draw.item_count <= 80:
            score += 0.3
        if draw.route_like:
            score -= 2.2
        scored.append((score, draw))

    if not scored:
        return {
            "marker_found": False,
            "marker_center": label_center,
            "marker_bbox": None,
            "snap_distance_pt": 0.0,
            "marker_score": 0.0,
            "cluster_count": 0,
            "marker_reason": "marker_not_found",
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 1.25:
        return {
            "marker_found": False,
            "marker_center": label_center,
            "marker_bbox": None,
            "snap_distance_pt": 0.0,
            "marker_score": round(best_score, 3),
            "cluster_count": 0,
            "marker_reason": "marker_score_low",
        }

    cluster = [draw for score, draw in scored if math.dist(best.center, draw.center) <= 8.0 and score >= best_score - 1.25]
    xs = [draw.center[0] for draw in cluster]
    ys = [draw.center[1] for draw in cluster]
    bbox = bbox_union([draw.bbox for draw in cluster])
    marker_center = (sum(xs) / len(xs), sum(ys) / len(ys))
    return {
        "marker_found": True,
        "marker_center": marker_center,
        "marker_bbox": bbox,
        "snap_distance_pt": round(math.dist(label_center, marker_center), 3),
        "marker_score": round(best_score, 3),
        "cluster_count": len(cluster),
        "marker_reason": "marker_snapped",
    }


def group_numbered_lines(
    lines_by_page: dict[int, list[LineRecord]],
    gazetteer_lookup: dict[tuple[str, int], GazetteerEntry],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    extracted_labels: list[dict[str, Any]] = []
    for page_no, lines in lines_by_page.items():
        for line in lines:
            extracted_labels.append(
                {
                    "label_id": line.label_id,
                    "page_no": line.page_no,
                    "frame_id": line.frame_id or "",
                    "text": line.text,
                    "role": line.role,
                    "color": line.color,
                    "color_hex": "#{:06x}".format(line.color),
                    "font_size": round(line.font_size, 3),
                    "bbox_x0_pt": round(line.bbox[0], 3),
                    "bbox_y0_pt": round(line.bbox[1], 3),
                    "bbox_x1_pt": round(line.bbox[2], 3),
                    "bbox_y1_pt": round(line.bbox[3], 3),
                }
            )

        numbered = [line for line in lines if line.role in {"temple_number", "bekkaku_number"}]
        for numbered_line in numbered:
            if numbered_line.role == "bekkaku_number":
                match = BEKKAKU_RE.search(numbered_line.text)
                temple_group = "bekkaku"
            else:
                match = NUMBER_RE.search(numbered_line.text)
                temple_group = "88"
            if not match:
                continue
            temple_no = int(match.group(1).translate(KANJI_DIGITS))
            if temple_group == "88" and not (1 <= temple_no <= 88):
                continue
            if temple_group == "bekkaku" and not (1 <= temple_no <= 20):
                continue
            is_okunoin = bool(match.group(2))
            gazetteer_entry = gazetteer_lookup.get((temple_group, temple_no))
            candidates: list[tuple[float, LineRecord]] = []
            numbered_center = numbered_line.center
            for other in lines:
                if other.label_id == numbered_line.label_id:
                    continue
                if other.frame_id != numbered_line.frame_id:
                    continue
                if other.role not in {"temple_name", "colored_label", "temple_reading"}:
                    continue
                dist = math.dist(numbered_center, other.center)
                if dist > 54.0:
                    continue
                y_gap = abs(other.center[1] - numbered_center[1])
                if y_gap > 38.0:
                    continue
                score = max(0.0, 36.0 - dist) / 12.0
                if other.color == numbered_line.color:
                    score += 1.6
                if other.role == "temple_name":
                    score += 2.0
                if other.role == "temple_reading":
                    score -= 2.0
                if other.font_size >= numbered_line.font_size + 1.0:
                    score += 0.8
                if TEMPLE_NAME_HINT_RE.search(other.text):
                    score += 1.0
                if other.text in {"奥之院", "奥の院"}:
                    score -= 3.0
                if other.role == "colored_label" and not TEMPLE_NAME_HINT_RE.search(other.text):
                    score -= 1.5
                if other.text.startswith("（"):
                    score -= 1.5
                if "解説編" in other.text:
                    score -= 2.0
                if gazetteer_entry:
                    sim, exact = similarity_score(other.text, gazetteer_entry)
                    score += sim * 2.4
                    if exact:
                        score += 1.4
                candidates.append((score, other))

            candidates.sort(key=lambda item: item[0], reverse=True)
            name_line = candidates[0][1] if candidates and candidates[0][0] > 1.0 else None

            reading_line: LineRecord | None = None
            if name_line:
                reading_candidates: list[tuple[float, LineRecord]] = []
                anchor = name_line.center
                for other in lines:
                    if other.label_id in {numbered_line.label_id, name_line.label_id}:
                        continue
                    if other.frame_id != numbered_line.frame_id:
                        continue
                    if other.role != "temple_reading":
                        continue
                    dist = math.dist(anchor, other.center)
                    if dist > 26.0:
                        continue
                    score = max(0.0, 24.0 - dist) / 10.0
                    if other.color == name_line.color:
                        score += 1.0
                    if other.font_size < name_line.font_size:
                        score += 0.3
                    reading_candidates.append((score, other))
                reading_candidates.sort(key=lambda item: item[0], reverse=True)
                if reading_candidates and reading_candidates[0][0] > 0.7:
                    reading_line = reading_candidates[0][1]

            boxes = [numbered_line.bbox]
            if name_line:
                boxes.append(name_line.bbox)
            if reading_line:
                boxes.append(reading_line.bbox)

            groups.append(
                {
                    "page_no": page_no,
                    "frame_id": numbered_line.frame_id,
                    "temple_group": temple_group,
                    "temple_no": temple_no,
                    "is_okunoin": is_okunoin,
                    "number_line": numbered_line,
                    "name_line": name_line,
                    "reading_line": reading_line,
                    "group_bbox": bbox_union(boxes),
                }
            )
    return groups, extracted_labels


def similarity_score(name: str, entry: GazetteerEntry) -> tuple[float, bool]:
    if not name:
        return (0.0, False)
    normalized = normalize_text(name)
    if normalized in entry.aliases:
        return (1.0, True)
    best = 0.0
    for alias in entry.aliases:
        best = max(best, SequenceMatcher(None, normalized, alias).ratio())
    return (best, False)


def build_gcp_candidates(
    groups: list[dict[str, Any]],
    gazetteer_rows: list[GazetteerEntry],
    draw_candidates_by_page: dict[int, list[DrawCandidate]],
    frames_by_page: dict[int, list[Frame]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gazetteer_index = {(row.temple_group, row.temple_no): row for row in gazetteer_rows}
    gcp_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    gcp_seq = 1

    for group in groups:
        page_no = group["page_no"]
        frame_id = group["frame_id"]
        name_line: LineRecord | None = group["name_line"]
        number_line: LineRecord = group["number_line"]
        reading_line: LineRecord | None = group["reading_line"]
        label_text = name_line.text if name_line else ""
        entry = gazetteer_index.get((group["temple_group"], group["temple_no"]))
        label_center = name_line.center if name_line else number_line.center
        marker = snap_to_marker(label_center, frame_id, draw_candidates_by_page.get(page_no, []))
        review_reasons: list[str] = []
        confidence = 0.0
        similarity = 0.0
        name_exact = False

        if frame_id is None:
            review_reasons.append("missing_frame_id")
        if group["is_okunoin"]:
            review_reasons.append("okunoin_label")
        if not marker["marker_found"]:
            review_reasons.append(marker["marker_reason"])

        if entry:
            similarity, name_exact = similarity_score(label_text, entry)
            confidence += 0.55
            if label_text:
                if name_exact:
                    confidence += 0.3
                    review_reasons.append("name_exact")
                else:
                    confidence += min(0.22, similarity * 0.22)
                    if similarity < 0.82:
                        review_reasons.append("name_similarity_low")
                    else:
                        review_reasons.append("name_fuzzy")
            else:
                review_reasons.append("name_missing")
            if marker["marker_found"]:
                confidence += 0.1
                review_reasons.append("marker_snapped")
            if number_line.color in TEMPLE_COLORS:
                confidence += 0.05
            if frame_id:
                confidence += 0.02
        else:
            review_reasons.append("gazetteer_missing")

        if marker["marker_found"] and marker["snap_distance_pt"] > 20:
            review_reasons.append("snap_distance_large")
            confidence -= 0.12
        if group["is_okunoin"]:
            confidence -= 0.4
        confidence = max(0.0, min(0.99, confidence))
        needs_manual_review = (
            confidence < 0.8
            or group["is_okunoin"]
            or not marker["marker_found"]
            or similarity < 0.9
            or frame_id is None
        )

        grouped_rows.append(
            {
                "page_no": page_no,
                "frame_id": frame_id or "",
                "temple_group": group["temple_group"],
                "temple_no": group["temple_no"],
                "is_okunoin": group["is_okunoin"],
                "number_text": number_line.text,
                "name_text": label_text,
                "reading_text": reading_line.text if reading_line else "",
                "label_x_pt": round(label_center[0], 3),
                "label_y_pt": round(label_center[1], 3),
                "marker_x_pt": round(marker["marker_center"][0], 3),
                "marker_y_pt": round(marker["marker_center"][1], 3),
                "marker_found": marker["marker_found"],
                "marker_score": marker["marker_score"],
                "snap_distance_pt": marker["snap_distance_pt"],
                "gazetteer_name": entry.name_short if entry else "",
                "name_similarity": round(similarity, 4),
                "confidence": round(confidence, 4),
                "needs_manual_review": needs_manual_review,
                "review_reasons": "|".join(dict.fromkeys(review_reasons)),
            }
        )

        if not entry or group["is_okunoin"] or not label_text or similarity < 0.6:
            if entry and not label_text:
                review_reasons.append("gcp_excluded_name_missing")
            if entry and label_text and similarity < 0.6:
                review_reasons.append("gcp_excluded_name_mismatch")
            unmatched_rows.append(
                {
                    "page_no": page_no,
                    "frame_id": frame_id or "",
                    "temple_group": group["temple_group"],
                    "temple_no": group["temple_no"],
                    "number_text": number_line.text,
                    "name_text": label_text,
                    "reading_text": reading_line.text if reading_line else "",
                    "label_x_pt": round(label_center[0], 3),
                    "label_y_pt": round(label_center[1], 3),
                    "marker_x_pt": round(marker["marker_center"][0], 3),
                    "marker_y_pt": round(marker["marker_center"][1], 3),
                    "review_reasons": "|".join(dict.fromkeys(review_reasons)),
                }
            )
            continue

        gcp_rows.append(
            {
                "gcp_id": f"gcp_{gcp_seq:04d}",
                "page_no": page_no,
                "frame_id": frame_id or "",
                "temple_group": entry.temple_group,
                "temple_no": entry.temple_no,
                "gazetteer_name_full": entry.name_full,
                "gazetteer_name_short": entry.name_short,
                "gazetteer_reading": entry.reading,
                "source_name_text": label_text,
                "source_number_text": number_line.text,
                "source_reading_text": reading_line.text if reading_line else "",
                "latitude": entry.latitude,
                "longitude": entry.longitude,
                "pdf_label_x_pt": round(label_center[0], 3),
                "pdf_label_y_pt": round(label_center[1], 3),
                "pdf_anchor_x_pt": round(marker["marker_center"][0], 3),
                "pdf_anchor_y_pt": round(marker["marker_center"][1], 3),
                "marker_found": marker["marker_found"],
                "snap_distance_pt": marker["snap_distance_pt"],
                "marker_score": marker["marker_score"],
                "name_similarity": round(similarity, 4),
                "confidence": round(confidence, 4),
                "needs_manual_review": needs_manual_review,
                "review_reasons": "|".join(dict.fromkeys(review_reasons)),
                "source_kind": "temple_number_name_match",
                "gazetteer_source": entry.source,
            }
        )
        gcp_seq += 1

    return gcp_rows, unmatched_rows, grouped_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_geojson(path: Path, rows: list[dict[str, Any]]) -> None:
    features = []
    for row in rows:
        properties = dict(row)
        lon = float(properties.pop("longitude"))
        lat = float(properties.pop("latitude"))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_preview_pages(gcp_rows: list[dict[str, Any]], unmatched_rows: list[dict[str, Any]]) -> list[int]:
    counts = Counter(row["page_no"] for row in gcp_rows)
    counts.update({row["page_no"]: 1 for row in unmatched_rows})
    return [page_no for page_no, _ in counts.most_common(12)]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def build_previews(
    doc: fitz.Document,
    preview_pages: list[int],
    grouped_rows: list[dict[str, Any]],
    gcp_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[int, str]:
    ensure_dir(out_dir)
    grouped_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    gcp_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_rows:
        grouped_by_page[int(row["page_no"])].append(row)
    for row in gcp_rows:
        gcp_by_page[int(row["page_no"])].append(row)

    preview_manifest: dict[int, str] = {}
    font = load_font(18)
    small_font = load_font(14)

    for page_no in preview_pages:
        page = doc[page_no - 1]
        matrix = fitz.Matrix(1.15, 1.15)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(image)

        for row in grouped_by_page.get(page_no, []):
            x = float(row["marker_x_pt"]) * 1.15
            y = float(row["marker_y_pt"]) * 1.15
            color = "#d44b3a" if row["needs_manual_review"] else "#1b7f5c"
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
            label = f"{row['temple_group']}-{row['temple_no']}"
            draw.text((x + 8, y - 10), label, fill=color, font=font)

        for row in gcp_by_page.get(page_no, []):
            x = float(row["pdf_anchor_x_pt"]) * 1.15
            y = float(row["pdf_anchor_y_pt"]) * 1.15
            draw.text((x + 8, y + 10), row["gcp_id"], fill="#124e66", font=small_font)

        out_path = out_dir / f"page_{page_no:03d}.jpg"
        image.save(out_path, quality=85)
        preview_manifest[page_no] = f"previews/{out_path.name}"
    return preview_manifest


def write_debug_bundle(
    out_dir: Path,
    gcp_geojson_path: Path,
    unmatched_rows: list[dict[str, Any]],
    preview_manifest: dict[int, str],
    token: str,
) -> None:
    debug_dir = out_dir / "mapbox_gcp_debug"
    ensure_dir(debug_dir)
    ensure_dir(debug_dir / "previews")

    geojson_target = debug_dir / "gcp_candidates.geojson"
    geojson_target.write_text(gcp_geojson_path.read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "unmatched_temple_labels.json").write_text(
        json.dumps(unmatched_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (debug_dir / "preview_manifest.json").write_text(
        json.dumps(preview_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for relative_path in preview_manifest.values():
        source = out_dir / relative_path
        target = debug_dir / relative_path
        ensure_dir(target.parent)
        if source.exists():
            shutil.copy2(source, target)
    html_path = debug_dir / "index.html"
    html_path.write_text(
        build_debug_html(token),
        encoding="utf-8",
    )


def build_debug_html(token: str) -> str:
    config = json.dumps({"mapboxToken": token}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Step 4 GCP Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffaf0;
      --line: #d6cbb6;
      --ink: #221f19;
      --accent: #0d6b6b;
      --warn: #b44d28;
    }}
    html, body {{
      margin: 0;
      height: 100%;
      background: var(--bg);
      color: var(--ink);
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
    }}
    #app {{
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100%;
    }}
    #sidebar {{
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 16px;
      overflow: auto;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    select, button {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 14px;
      margin-bottom: 10px;
    }}
    .metric {{
      margin: 10px 0;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: white;
      border-radius: 10px;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
    }}
    .preview {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      margin-top: 10px;
    }}
    .list {{
      margin-top: 12px;
      font-size: 12px;
      line-height: 1.5;
    }}
    .item {{
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      margin-bottom: 8px;
    }}
    .warn {{
      color: var(--warn);
      font-weight: 700;
    }}
    .ok {{
      color: var(--accent);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h1>Step 4 GCP Debug</h1>
      <select id="pageSelect"></select>
      <button id="reviewToggle">レビュー対象のみ: OFF</button>
      <div class="metric"><strong id="candidateCount">0</strong>GCP candidates</div>
      <div class="metric"><strong id="unmatchedCount">0</strong>unmatched temple labels</div>
      <img id="preview" class="preview" alt="page preview" />
      <div id="candidateList" class="list"></div>
      <div id="unmatchedList" class="list"></div>
    </aside>
    <main id="map"></main>
  </div>
  <script>window.DEBUG_CONFIG = {config};</script>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js"></script>
  <script>
    const blankStyle = {{
      version: 8,
      sources: {{}},
      layers: [{{ id: 'background', type: 'background', paint: {{ 'background-color': '#f6f2ea' }} }}]
    }};

    const token = window.DEBUG_CONFIG.mapboxToken || '';
    if (token) {{
      mapboxgl.accessToken = token;
    }}

    const state = {{
      page: null,
      reviewOnly: false,
      features: [],
      unmatched: [],
      previews: {{}},
    }};

    const els = {{
      pageSelect: document.getElementById('pageSelect'),
      reviewToggle: document.getElementById('reviewToggle'),
      candidateCount: document.getElementById('candidateCount'),
      unmatchedCount: document.getElementById('unmatchedCount'),
      preview: document.getElementById('preview'),
      candidateList: document.getElementById('candidateList'),
      unmatchedList: document.getElementById('unmatchedList'),
    }};

    const map = new mapboxgl.Map({{
      container: 'map',
      style: token ? 'mapbox://styles/mapbox/outdoors-v12' : blankStyle,
      center: [133.6, 33.8],
      zoom: 7.2
    }});

    async function loadData() {{
      const [gcpRes, unmatchedRes, previewsRes] = await Promise.all([
        fetch('gcp_candidates.geojson'),
        fetch('unmatched_temple_labels.json'),
        fetch('preview_manifest.json')
      ]);
      state.features = (await gcpRes.json()).features;
      state.unmatched = await unmatchedRes.json();
      state.previews = await previewsRes.json();
      initSelectors();
      render();
    }}

    function initSelectors() {{
      const pages = Array.from(new Set([
        ...state.features.map(f => f.properties.page_no),
        ...state.unmatched.map(row => row.page_no),
      ])).sort((a, b) => a - b);
      state.page = pages[0] || null;
      els.pageSelect.innerHTML = '';
      for (const page of pages) {{
        const opt = document.createElement('option');
        opt.value = String(page);
        opt.textContent = `Page ${{page}}`;
        els.pageSelect.appendChild(opt);
      }}
      els.pageSelect.addEventListener('change', () => {{
        state.page = Number(els.pageSelect.value);
        render();
      }});
      els.reviewToggle.addEventListener('click', () => {{
        state.reviewOnly = !state.reviewOnly;
        els.reviewToggle.textContent = `レビュー対象のみ: ${{state.reviewOnly ? 'ON' : 'OFF'}}`;
        render();
      }});
    }}

    function currentFeatures() {{
      return state.features.filter(feature => {{
        if (feature.properties.page_no !== state.page) return false;
        if (state.reviewOnly && !feature.properties.needs_manual_review) return false;
        return true;
      }});
    }}

    function currentUnmatched() {{
      return state.unmatched.filter(row => row.page_no === state.page);
    }}

    function ensureLayers() {{
      if (!map.getSource('gcp')) {{
        map.addSource('gcp', {{ type: 'geojson', data: {{ type: 'FeatureCollection', features: [] }} }});
        map.addLayer({{
          id: 'gcp-circle',
          type: 'circle',
          source: 'gcp',
          paint: {{
            'circle-radius': [
              'interpolate', ['linear'], ['get', 'confidence'],
              0.5, 6,
              0.95, 12
            ],
            'circle-color': [
              'case',
              ['boolean', ['get', 'needs_manual_review'], false], '#c55a11',
              '#157f63'
            ],
            'circle-stroke-color': '#fff',
            'circle-stroke-width': 1.5
          }}
        }});
        map.addLayer({{
          id: 'gcp-label',
          type: 'symbol',
          source: 'gcp',
          layout: {{
            'text-field': ['to-string', ['get', 'temple_no']],
            'text-size': 11,
            'text-offset': [0, 1.4]
          }},
          paint: {{
            'text-color': '#1f1a16',
            'text-halo-color': '#fff8ef',
            'text-halo-width': 1.2
          }}
        }});
        map.on('click', 'gcp-circle', event => {{
          const feature = event.features?.[0];
          if (!feature) return;
          const p = feature.properties;
          new mapboxgl.Popup()
            .setLngLat(feature.geometry.coordinates)
            .setHTML(`<strong>${{p.gcp_id}}</strong><br>${{p.gazetteer_name_short}}<br>page ${{p.page_no}} / frame ${{p.frame_id || '-'}}<br>confidence=${{p.confidence}}<br>${{p.review_reasons}}`)
            .addTo(map);
        }});
      }}
    }}

    function render() {{
      const features = currentFeatures();
      const unmatched = currentUnmatched();
      els.candidateCount.textContent = String(features.length);
      els.unmatchedCount.textContent = String(unmatched.length);
      const previewPath = state.previews[String(state.page)] || state.previews[state.page];
      if (previewPath) {{
        els.preview.src = previewPath;
        els.preview.style.display = 'block';
      }} else {{
        els.preview.style.display = 'none';
      }}

      els.candidateList.innerHTML = features.map(feature => {{
        const p = feature.properties;
        return `<div class="item"><div class="${{p.needs_manual_review ? 'warn' : 'ok'}}">${{p.gcp_id}} / ${{p.gazetteer_name_short}}</div><div>confidence: ${{p.confidence}}</div><div>similarity: ${{p.name_similarity}}</div><div>${{p.review_reasons}}</div></div>`;
      }}).join('') || '<div class="item">候補なし</div>';

      els.unmatchedList.innerHTML = unmatched.map(row => {{
        return `<div class="item"><div class="warn">${{row.temple_group}}-${{row.temple_no}}</div><div>${{row.name_text || row.number_text}}</div><div>${{row.review_reasons}}</div></div>`;
      }}).join('') || '';

      if (!map.isStyleLoaded()) return;
      ensureLayers();
      const source = map.getSource('gcp');
      source.setData({{ type: 'FeatureCollection', features }});
      if (features.length) {{
        const bounds = new mapboxgl.LngLatBounds();
        for (const feature of features) bounds.extend(feature.geometry.coordinates);
        map.fitBounds(bounds, {{ padding: 48, maxZoom: 12 }});
      }}
    }}

    map.on('load', () => {{
      ensureLayers();
      render();
    }});

    loadData();
  </script>
</body>
</html>
"""


def write_report(
    path: Path,
    pdf_name: str,
    gazetteer_rows: list[GazetteerEntry],
    groups: list[dict[str, Any]],
    gcp_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
    extracted_labels: list[dict[str, Any]],
) -> None:
    role_counts = Counter(row["role"] for row in extracted_labels)
    group_counts = Counter(row["temple_group"] for row in gcp_rows)
    review_count = sum(1 for row in gcp_rows if row["needs_manual_review"])
    average_confidence = sum(float(row["confidence"]) for row in gcp_rows) / max(len(gcp_rows), 1)
    lines = [
        "# Step 4 Report",
        "",
        f"- PDF: `{pdf_name}`",
        f"- Gazetteer rows: `{len(gazetteer_rows)}`",
        f"- Numbered temple groups: `{len(groups)}`",
        f"- Matched GCP candidates: `{len(gcp_rows)}`",
        f"- Unmatched / excluded temple labels: `{len(unmatched_rows)}`",
        f"- Manual review GCPs: `{review_count}`",
        f"- Average confidence: `{average_confidence:.3f}`",
        "",
        "## Label Counts",
    ]
    for key, value in sorted(role_counts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## GCP Coverage",
        ]
    )
    for key, value in sorted(group_counts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "- Gazetteer: `artifacts/step4/gazetteer.csv`",
            "- Extracted labels: `artifacts/step4/extracted_labels.csv`",
            "- Temple label groups: `artifacts/step4/temple_label_groups.csv`",
            "- GCP candidates CSV: `artifacts/step4/gcp_candidates.csv`",
            "- GCP candidates GeoJSON: `artifacts/step4/gcp_candidates.geojson`",
            "- Unmatched temple labels: `artifacts/step4/unmatched_temple_labels.csv`",
            "- Mapbox debug HTML: `artifacts/step4/mapbox_gcp_debug/index.html`",
            "",
            "## Notes",
            "- `okunoin_label` は主札所の座標に直接結びつかないので GCP から除外しています。",
            "- `marker_not_found` / `marker_score_low` は PDF 上の寺院マーカー snap が弱かった箇所です。",
            "- `name_similarity_low` は番号一致で候補化できたものの、寺院名照合が弱い箇所です。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: extract GCP candidates from temple labels and gazetteer")
    parser.add_argument(
        "--pdf",
        default="四国遍路ひとり歩き同行二人（地図編）第14版第1刷.pdf",
        type=Path,
    )
    parser.add_argument(
        "--frames",
        default=Path("artifacts/step2/frames.csv"),
        type=Path,
    )
    parser.add_argument(
        "--gazetteer-88",
        default=Path("data_sources/step4/88.csv"),
        type=Path,
    )
    parser.add_argument(
        "--gazetteer-bekkaku",
        default=Path("data_sources/step4/bekkaku20_portal.html"),
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        default=Path("artifacts/step4"),
        type=Path,
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / "previews")

    frames_by_page = load_frames(args.frames)
    gazetteer_rows = build_88_gazetteer(args.gazetteer_88) + build_bekkaku_gazetteer(args.gazetteer_bekkaku)
    gazetteer_lookup = {(row.temple_group, row.temple_no): row for row in gazetteer_rows}
    doc = fitz.open(args.pdf)
    line_records, lines_by_page = extract_lines(doc, frames_by_page)
    groups, extracted_labels = group_numbered_lines(lines_by_page, gazetteer_lookup)
    draw_candidates_by_page = {page_no: build_draw_candidates(doc[page_no - 1]) for page_no in lines_by_page}
    gcp_rows, unmatched_rows, grouped_rows = build_gcp_candidates(groups, gazetteer_rows, draw_candidates_by_page, frames_by_page)

    gazetteer_csv_rows = [
        {
            "temple_group": row.temple_group,
            "temple_no": row.temple_no,
            "name_full": row.name_full,
            "name_short": row.name_short,
            "reading": row.reading,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "source": row.source,
            "aliases": "|".join(sorted(row.aliases)),
        }
        for row in gazetteer_rows
    ]

    write_csv(args.out_dir / "gazetteer.csv", gazetteer_csv_rows)
    write_csv(args.out_dir / "extracted_labels.csv", extracted_labels)
    write_csv(args.out_dir / "temple_label_groups.csv", grouped_rows)
    write_csv(args.out_dir / "gcp_candidates.csv", gcp_rows)
    write_csv(args.out_dir / "unmatched_temple_labels.csv", unmatched_rows)
    write_geojson(args.out_dir / "gcp_candidates.geojson", gcp_rows)

    preview_pages = pick_preview_pages(gcp_rows, unmatched_rows)
    preview_manifest = build_previews(doc, preview_pages, grouped_rows, gcp_rows, args.out_dir / "previews")
    token = read_mapbox_token()
    write_debug_bundle(args.out_dir, args.out_dir / "gcp_candidates.geojson", unmatched_rows, preview_manifest, token)

    extraction_log = {
        "pdf": str(args.pdf),
        "page_count": doc.page_count,
        "gazetteer_rows": len(gazetteer_rows),
        "label_count": len(line_records),
        "numbered_group_count": len(groups),
        "gcp_candidate_count": len(gcp_rows),
        "unmatched_group_count": len(unmatched_rows),
        "manual_review_count": sum(1 for row in gcp_rows if row["needs_manual_review"]),
        "preview_pages": preview_pages,
        "mapbox_token_present": bool(token),
    }
    (args.out_dir / "extraction_log.json").write_text(
        json.dumps(extraction_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        args.out_dir / "step4_report.md",
        args.pdf.name,
        gazetteer_rows,
        groups,
        gcp_rows,
        unmatched_rows,
        extracted_labels,
    )


if __name__ == "__main__":
    main()
