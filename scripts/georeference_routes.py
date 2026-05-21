#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAPBOX_SCALE = 0.01
EARTH_RADIUS_M = 6378137.0


@dataclass
class GCP:
    gcp_id: str
    page_no: int
    frame_id: str
    temple_group: str
    temple_no: int
    pdf_x: float
    pdf_y: float
    latitude: float
    longitude: float
    confidence: float
    needs_manual_review: bool
    page_height_pt: float
    source: str


@dataclass
class TransformModel:
    scope_type: str
    scope_id: str
    page_no: int
    frame_id: str | None
    model_name: str
    y_mode: str
    crs_candidate: str
    min_points: int
    ref_lon: float
    ref_lat: float
    params: Any
    gcp_ids: list[str]
    gcp_count: int
    manual_count: int
    auto_count: int
    gcp_mean_confidence: float
    rmse_m: float
    loocv_rmse_m: float | None
    quality_status: str
    seam_err_m: float | None = None
    duplicate_count: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_mapbox_token() -> str:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MAPBOX_ACCESS_TOKEN="):
                return line.partition("=")[2].strip()
    return ""


def load_page_heights(path: Path) -> dict[int, float]:
    heights: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("page_no") or not row.get("page_height_pt"):
                continue
            heights[int(row["page_no"])] = float(row["page_height_pt"])
    return heights


def load_gcps(path: Path, page_heights: dict[int, float]) -> list[GCP]:
    rows: list[GCP] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            page_no = int(row["page_no"])
            page_height_pt = float(row["page_height_pt"]) if row.get("page_height_pt") else page_heights.get(page_no)
            if page_height_pt is None:
                raise SystemExit(f"page_height_pt が見つかりません: page {page_no}")
            rows.append(
                GCP(
                    gcp_id=row["gcp_id"],
                    page_no=page_no,
                    frame_id=row["frame_id"],
                    temple_group=row["temple_group"],
                    temple_no=int(row["temple_no"]),
                    pdf_x=float(row["pdf_anchor_x_pt"]),
                    pdf_y=float(row["pdf_anchor_y_pt"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    confidence=float(row["confidence"]),
                    needs_manual_review=row["needs_manual_review"] == "True",
                    page_height_pt=float(page_height_pt),
                    source=row.get("source", "auto"),
                )
            )
    return rows


def load_routes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data["features"]


def synthetic_to_pdf(coord: list[float]) -> tuple[float, float]:
    return (coord[0] / MAPBOX_SCALE, -coord[1] / MAPBOX_SCALE)


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon)
    lat = max(-85.05112878, min(85.05112878, lat))
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return (x, y)


def web_mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2.0 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2.0)
    return (lon, lat)


def lonlat_to_local_equirect(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return (x, y)


def local_equirect_to_lonlat(x: float, y: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    return (lon, lat)


def target_xy(gcps: list[GCP], crs_candidate: str, ref_lon: float, ref_lat: float) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for gcp in gcps:
        if crs_candidate == "web_mercator_m":
            points.append(lonlat_to_web_mercator(gcp.longitude, gcp.latitude))
        else:
            points.append(lonlat_to_local_equirect(gcp.longitude, gcp.latitude, ref_lon, ref_lat))
    return np.array(points, dtype=float)


def xy_to_lonlat(points: np.ndarray, crs_candidate: str, ref_lon: float, ref_lat: float) -> np.ndarray:
    lonlat: list[tuple[float, float]] = []
    for x, y in points:
        if crs_candidate == "web_mercator_m":
            lonlat.append(web_mercator_to_lonlat(float(x), float(y)))
        else:
            lonlat.append(local_equirect_to_lonlat(float(x), float(y), ref_lon, ref_lat))
    return np.array(lonlat, dtype=float)


def fit_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
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
    if variance <= 1e-12:
        raise np.linalg.LinAlgError("degenerate source variance")
    scale = np.trace(np.diag(d) @ s) / variance
    translation = tgt_mean - (scale * (rotation @ src_mean))
    return {"rotation": rotation, "scale": scale, "translation": translation}


def apply_similarity(points: np.ndarray, params: dict[str, np.ndarray | float]) -> np.ndarray:
    rotation = params["rotation"]
    scale = float(params["scale"])
    translation = params["translation"]
    return ((scale * (rotation @ points.T)).T) + translation


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([source[:, 0], source[:, 1], np.ones(len(source))])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return params


def apply_affine(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    return design @ params


def fit_projective(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    matrix = np.array(rows, dtype=float)
    rhs = np.array(values, dtype=float)
    params, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return params


def apply_projective(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    h11, h12, h13, h21, h22, h23, h31, h32 = params
    x = points[:, 0]
    y = points[:, 1]
    denom = (h31 * x) + (h32 * y) + 1.0
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    u = ((h11 * x) + (h12 * y) + h13) / denom
    v = ((h21 * x) + (h22 * y) + h23) / denom
    return np.column_stack([u, v])


def fit_poly2(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = source[:, 0]
    y = source[:, 1]
    design = np.column_stack([np.ones(len(source)), x, y, x * x, x * y, y * y])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return params


def apply_poly2(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([np.ones(len(points)), x, y, x * x, x * y, y * y])
    return design @ params


MODEL_SPECS: dict[str, tuple[int, Any, Any]] = {
    "similarity": (2, fit_similarity, apply_similarity),
    "affine": (3, fit_affine, apply_affine),
    "projective": (4, fit_projective, apply_projective),
    "polynomial2": (8, fit_poly2, apply_poly2),
}


def rms_errors(left: np.ndarray, right: np.ndarray) -> float:
    deltas = left - right
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def source_points(gcps: list[GCP], y_mode: str) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for gcp in gcps:
        y = gcp.pdf_y if y_mode == "raw_y" else (gcp.page_height_pt - gcp.pdf_y)
        rows.append((gcp.pdf_x, y))
    return np.array(rows, dtype=float)


def model_quality_status(gcp_count: int, loocv_rmse_m: float | None) -> str:
    if gcp_count < 4:
        return "fail"
    if loocv_rmse_m is None:
        return "fail"
    if loocv_rmse_m < 50.0:
        return "pass"
    if loocv_rmse_m <= 150.0:
        return "review"
    return "fail"


def gcps_for_scope(scope_gcps: list[GCP], scope_type: str) -> list[GCP]:
    manual_gcps = [gcp for gcp in scope_gcps if gcp.source == "manual"]
    if scope_type == "frame" and manual_gcps:
        return manual_gcps
    return scope_gcps


def fit_candidate_model(
    gcps: list[GCP],
    scope_type: str,
    scope_id: str,
    page_no: int,
    frame_id: str | None,
    model_name: str,
    y_mode: str,
    crs_candidate: str,
) -> TransformModel | None:
    min_points, fit_fn, apply_fn = MODEL_SPECS[model_name]
    used_gcps = gcps_for_scope(gcps, scope_type)
    if len(used_gcps) < min_points:
        return None
    source = source_points(used_gcps, y_mode)
    ref_lon = sum(gcp.longitude for gcp in used_gcps) / len(used_gcps)
    ref_lat = sum(gcp.latitude for gcp in used_gcps) / len(used_gcps)
    target = target_xy(used_gcps, crs_candidate, ref_lon, ref_lat)
    try:
        params = fit_fn(source, target)
        fitted = apply_fn(source, params)
    except np.linalg.LinAlgError:
        return None
    rmse = rms_errors(fitted, target)

    loocv_errors: list[float] = []
    if len(used_gcps) > min_points:
        for index in range(len(used_gcps)):
            train_idx = [i for i in range(len(used_gcps)) if i != index]
            train_source = source[train_idx]
            train_target = target[train_idx]
            try:
                train_params = fit_fn(train_source, train_target)
                predicted = apply_fn(source[index : index + 1], train_params)
            except np.linalg.LinAlgError:
                continue
            loocv_errors.append(float(np.linalg.norm(predicted[0] - target[index])))
    loocv_rmse = float(np.sqrt(np.mean(np.square(loocv_errors)))) if loocv_errors else None
    quality_status = model_quality_status(len(used_gcps), loocv_rmse)
    manual_count = sum(1 for gcp in used_gcps if gcp.source == "manual")
    auto_count = sum(1 for gcp in used_gcps if gcp.source != "manual")

    return TransformModel(
        scope_type=scope_type,
        scope_id=scope_id,
        page_no=page_no,
        frame_id=frame_id,
        model_name=model_name,
        y_mode=y_mode,
        crs_candidate=crs_candidate,
        min_points=min_points,
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        params=params,
        gcp_ids=[gcp.gcp_id for gcp in used_gcps],
        gcp_count=len(used_gcps),
        manual_count=manual_count,
        auto_count=auto_count,
        gcp_mean_confidence=sum(gcp.confidence for gcp in used_gcps) / len(used_gcps),
        rmse_m=rmse,
        loocv_rmse_m=loocv_rmse,
        quality_status=quality_status,
    )


def model_sort_key(model: TransformModel) -> tuple[float, float, float]:
    loocv = model.loocv_rmse_m if model.loocv_rmse_m is not None else (model.rmse_m + 1000.0)
    complexity_penalty = {"similarity": 0.0, "affine": 5.0, "projective": 15.0, "polynomial2": 30.0}[model.model_name]
    y_penalty = {"y_flipped": 0.0, "raw_y": 60.0}[model.y_mode]
    status_penalty = {"pass": 0.0, "review": 25.0, "fail": 1000.0}[model.quality_status]
    return (loocv + complexity_penalty + y_penalty + status_penalty, model.rmse_m, -model.gcp_count)


def candidate_metric(model: TransformModel) -> float:
    return model.loocv_rmse_m if model.loocv_rmse_m is not None else (model.rmse_m + 1000.0)


def alternative_clearly_better(candidate: TransformModel, baseline: TransformModel) -> bool:
    if candidate.model_name == "similarity":
        return False
    improvement = candidate_metric(baseline) - candidate_metric(candidate)
    required = max(15.0, candidate_metric(baseline) * 0.2)
    return improvement >= required


def select_best_models(gcps: list[GCP]) -> tuple[dict[str, TransformModel], list[dict[str, Any]]]:
    by_scope: dict[str, list[GCP]] = defaultdict(list)
    scope_meta: dict[str, tuple[str, int, str | None]] = {}
    for gcp in gcps:
        if gcp.frame_id:
            scope = f"frame:{gcp.page_no}:{gcp.frame_id}"
            by_scope[scope].append(gcp)
            scope_meta[scope] = ("frame", gcp.page_no, gcp.frame_id)
        page_scope = f"page:{gcp.page_no}"
        by_scope[page_scope].append(gcp)
        scope_meta[page_scope] = ("page", gcp.page_no, None)

    selected: dict[str, TransformModel] = {}
    diagnostics: list[dict[str, Any]] = []
    for scope_id, scope_gcps in by_scope.items():
        scope_type, page_no, frame_id = scope_meta[scope_id]
        candidates: list[TransformModel] = []
        for y_mode in ["y_flipped", "raw_y"]:
            for crs_candidate in ["local_equirect_m", "web_mercator_m"]:
                for model_name in MODEL_SPECS:
                    fitted = fit_candidate_model(
                        scope_gcps,
                        scope_type=scope_type,
                        scope_id=scope_id,
                        page_no=page_no,
                        frame_id=frame_id,
                        model_name=model_name,
                        y_mode=y_mode,
                        crs_candidate=crs_candidate,
                    )
                    if not fitted:
                        continue
                    candidates.append(fitted)
                    diagnostics.append(
                        {
                            "scope_id": scope_id,
                            "scope_type": scope_type,
                            "page_no": page_no,
                            "frame_id": frame_id or "",
                            "candidate_model": model_name,
                            "y_mode": y_mode,
                            "crs_candidate": crs_candidate,
                            "gcp_count": fitted.gcp_count,
                            "manual_count": fitted.manual_count,
                            "auto_count": fitted.auto_count,
                            "gcp_mean_confidence": round(fitted.gcp_mean_confidence, 4),
                            "rmse_m": round(fitted.rmse_m, 3),
                            "loocv_rmse_m": "" if fitted.loocv_rmse_m is None else round(fitted.loocv_rmse_m, 3),
                            "quality_status": fitted.quality_status,
                            "selected": False,
                        }
                    )
        eligible = [candidate for candidate in candidates if candidate.quality_status in {"pass", "review"}]
        if not eligible:
            continue

        similarity_candidates = [candidate for candidate in eligible if candidate.model_name == "similarity"]
        base_model = sorted(similarity_candidates, key=model_sort_key)[0] if similarity_candidates else sorted(eligible, key=model_sort_key)[0]
        selected_model = base_model
        for candidate in sorted(eligible, key=model_sort_key):
            if candidate is base_model:
                continue
            if alternative_clearly_better(candidate, base_model):
                selected_model = candidate
                break
        selected[scope_id] = selected_model
        for row in diagnostics:
            if (
                row["scope_id"] == scope_id
                and row["candidate_model"] == selected_model.model_name
                and row["y_mode"] == selected_model.y_mode
                and row["crs_candidate"] == selected_model.crs_candidate
            ):
                row["selected"] = True
                break
    return selected, diagnostics


def apply_model(model: TransformModel, pdf_points: np.ndarray) -> np.ndarray:
    _, _, apply_fn = MODEL_SPECS[model.model_name]
    xy = apply_fn(pdf_points, model.params)
    return xy_to_lonlat(xy, model.crs_candidate, model.ref_lon, model.ref_lat)


def compute_seam_errors(models: dict[str, TransformModel], gcps: list[GCP]) -> None:
    gcps_by_scope: dict[str, list[GCP]] = defaultdict(list)
    for gcp in gcps:
        if gcp.frame_id:
            gcps_by_scope[f"frame:{gcp.page_no}:{gcp.frame_id}"].append(gcp)
        gcps_by_scope[f"page:{gcp.page_no}"].append(gcp)

    duplicates: dict[tuple[str, int], list[tuple[str, GCP]]] = defaultdict(list)
    for scope_id, model in models.items():
        if model.scope_type != "frame":
            continue
        for gcp in gcps_by_scope.get(scope_id, []):
            duplicates[(gcp.temple_group, gcp.temple_no)].append((scope_id, gcp))

    seam_by_scope: dict[str, list[float]] = defaultdict(list)
    for dup_items in duplicates.values():
        if len(dup_items) < 2:
            continue
        transformed: list[tuple[str, np.ndarray]] = []
        for scope_id, gcp in dup_items:
            model = models.get(scope_id)
            if not model:
                continue
            points = source_points([gcp], model.y_mode)
            lonlat = apply_model(model, points)[0]
            transformed.append((scope_id, lonlat))
        for i in range(len(transformed)):
            for j in range(i + 1, len(transformed)):
                lonlat_a = transformed[i][1]
                lonlat_b = transformed[j][1]
                dist = haversine_m(lonlat_a[1], lonlat_a[0], lonlat_b[1], lonlat_b[0])
                seam_by_scope[transformed[i][0]].append(dist)
                seam_by_scope[transformed[j][0]].append(dist)

    for scope_id, values in seam_by_scope.items():
        model = models[scope_id]
        model.seam_err_m = sum(values) / len(values)
        model.duplicate_count = len(values)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def choose_route_model(
    route: dict[str, Any],
    models: dict[str, TransformModel],
    *,
    allow_page_fallback: bool,
) -> tuple[TransformModel | None, list[str]]:
    page_no = int(route["properties"]["page_no"])
    frame_id = route["properties"].get("frame_id") or ""
    reasons: list[str] = []
    if frame_id:
        scope_id = f"frame:{page_no}:{frame_id}"
        if scope_id in models:
            return models[scope_id], reasons
        reasons.append("missing_frame_model")
        if not allow_page_fallback:
            reasons.append("page_model_fallback_disabled")
            return None, reasons
    page_scope = f"page:{page_no}"
    if page_scope in models:
        reasons.append("page_model_fallback")
        return models[page_scope], reasons
    reasons.append("missing_page_model")
    return None, reasons


def source_points_for_geometry(geometry: dict[str, Any], y_mode: str, page_height_pt: float) -> list[np.ndarray]:
    if geometry["type"] == "LineString":
        raw = np.array([synthetic_to_pdf(coord) for coord in geometry["coordinates"]], dtype=float)
        if y_mode == "y_flipped":
            raw[:, 1] = page_height_pt - raw[:, 1]
        return [raw]
    if geometry["type"] == "MultiLineString":
        lines: list[np.ndarray] = []
        for line in geometry["coordinates"]:
            raw = np.array([synthetic_to_pdf(coord) for coord in line], dtype=float)
            if y_mode == "y_flipped":
                raw[:, 1] = page_height_pt - raw[:, 1]
            lines.append(raw)
        return lines
    raise ValueError(f"unsupported geometry type: {geometry['type']}")


def transform_geometry(geometry: dict[str, Any], model: TransformModel, page_height_pt: float) -> dict[str, Any]:
    if geometry["type"] == "LineString":
        pdf_points = source_points_for_geometry(geometry, model.y_mode, page_height_pt)[0]
        lonlat = apply_model(model, pdf_points)
        return {
            "type": "LineString",
            "coordinates": [[round(float(lon), 7), round(float(lat), 7)] for lon, lat in lonlat],
        }
    if geometry["type"] == "MultiLineString":
        lines = []
        for pdf_points in source_points_for_geometry(geometry, model.y_mode, page_height_pt):
            lonlat = apply_model(model, pdf_points)
            lines.append([[round(float(lon), 7), round(float(lat), 7)] for lon, lat in lonlat])
        return {"type": "MultiLineString", "coordinates": lines}
    raise ValueError(f"unsupported geometry type: {geometry['type']}")


def geometry_coordinate_iter(geometry: dict[str, Any]) -> list[list[float]]:
    if geometry["type"] == "LineString":
        return geometry["coordinates"]
    coords: list[list[float]] = []
    for line in geometry["coordinates"]:
        coords.extend(line)
    return coords


def geometry_is_plausible_shikoku(geometry: dict[str, Any]) -> bool:
    coords = geometry_coordinate_iter(geometry)
    if not coords:
        return False
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    if min(lons) < 131.0 or max(lons) > 135.6:
        return False
    if min(lats) < 32.0 or max(lats) > 35.2:
        return False
    if (max(lons) - min(lons)) > 1.8:
        return False
    if (max(lats) - min(lats)) > 1.5:
        return False
    return True


def route_transform_confidence(route_conf: float, model: TransformModel, used_page_fallback: bool) -> float:
    score = route_conf
    score *= model.gcp_mean_confidence
    if model.loocv_rmse_m is not None:
        if model.loocv_rmse_m < 50.0:
            score *= 0.98
        elif model.loocv_rmse_m <= 150.0:
            score *= max(0.45, 0.8 - ((model.loocv_rmse_m - 50.0) / 250.0))
        else:
            score *= 0.2
    else:
        score *= 0.2
    if model.seam_err_m is not None:
        score *= max(0.3, 1.0 - (model.seam_err_m / 2000.0))
    if used_page_fallback:
        score *= 0.82
    return max(0.0, min(0.99, score))


def build_outputs(
    routes: list[dict[str, Any]],
    models: dict[str, TransformModel],
    page_heights: dict[int, float],
    *,
    allow_page_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    transformed: list[dict[str, Any]] = []
    untransformed: list[dict[str, Any]] = []
    route_status: list[dict[str, Any]] = []

    for feature in routes:
        properties = dict(feature["properties"])
        route_id = properties["route_id"]
        model, reasons = choose_route_model(feature, models, allow_page_fallback=allow_page_fallback)
        page_no = int(properties["page_no"])
        frame_id = properties.get("frame_id") or ""
        page_height_pt = page_heights.get(page_no)
        route_conf = float(properties.get("confidence") or 0.0)
        existing_reasons = [reason for reason in str(properties.get("review_reasons") or "").split(",") if reason]
        needs_manual_review = bool(properties.get("needs_manual_review"))

        if model is None:
            status_reasons = existing_reasons + reasons + ["route_not_transformed"]
            route_status.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "transform_scope": "",
                    "transform_model": "",
                    "y_mode": "",
                    "model_quality_status": "",
                    "crs_candidate": "",
                    "gcp_count": 0,
                    "rmse_m": "",
                    "loocv_rmse_m": "",
                    "seam_err_m": "",
                    "confidence": round(route_conf, 4),
                    "needs_manual_review": True,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons)),
                }
            )
            untransformed.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons)),
                    "style_class": properties.get("style_class", ""),
                }
            )
            continue

        if page_height_pt is None:
            status_reasons = existing_reasons + reasons + ["missing_page_height", "route_not_transformed"]
            route_status.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "transform_scope": model.scope_id,
                    "transform_model": model.model_name,
                    "y_mode": model.y_mode,
                    "model_quality_status": model.quality_status,
                    "crs_candidate": model.crs_candidate,
                    "gcp_count": model.gcp_count,
                    "rmse_m": round(model.rmse_m, 3),
                    "loocv_rmse_m": "" if model.loocv_rmse_m is None else round(model.loocv_rmse_m, 3),
                    "seam_err_m": "" if model.seam_err_m is None else round(model.seam_err_m, 3),
                    "confidence": round(route_conf, 4),
                    "needs_manual_review": True,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons)),
                }
            )
            untransformed.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons)),
                    "style_class": properties.get("style_class", ""),
                }
            )
            continue

        used_page_fallback = model.scope_type == "page" and bool(frame_id)
        transform_conf = route_transform_confidence(route_conf, model, used_page_fallback)
        status_reasons = existing_reasons + reasons
        if used_page_fallback:
            status_reasons.append("page_model_fallback")
        if model.quality_status == "review":
            status_reasons.append("loocv_review")
        if model.quality_status == "fail":
            status_reasons.append("loocv_fail")
        if model.rmse_m > 150:
            status_reasons.append("fit_rmse_high")
        if model.seam_err_m is not None and model.seam_err_m > 200:
            status_reasons.append("seam_proxy_high")

        try:
            geometry = transform_geometry(feature["geometry"], model, page_height_pt)
        except Exception:
            status_reasons.append("transform_failed")
            route_status.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "transform_scope": model.scope_id,
                    "transform_model": model.model_name,
                    "y_mode": model.y_mode,
                    "model_quality_status": model.quality_status,
                    "crs_candidate": model.crs_candidate,
                    "gcp_count": model.gcp_count,
                    "rmse_m": round(model.rmse_m, 3),
                    "loocv_rmse_m": "" if model.loocv_rmse_m is None else round(model.loocv_rmse_m, 3),
                    "seam_err_m": "" if model.seam_err_m is None else round(model.seam_err_m, 3),
                    "confidence": round(transform_conf, 4),
                    "needs_manual_review": True,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons)),
                }
            )
            continue

        if not geometry_is_plausible_shikoku(geometry):
            status_reasons.append("geographic_bounds_failed")
            route_status.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "transform_scope": model.scope_id,
                    "transform_model": "",
                    "y_mode": "",
                    "model_quality_status": "",
                    "crs_candidate": "",
                    "gcp_count": 0,
                    "rmse_m": "",
                    "loocv_rmse_m": "",
                    "seam_err_m": "",
                    "confidence": round(route_conf, 4),
                    "needs_manual_review": True,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons + ["route_not_transformed"])),
                }
            )
            untransformed.append(
                {
                    "route_id": route_id,
                    "page_no": page_no,
                    "frame_id": frame_id,
                    "review_reasons": ",".join(dict.fromkeys(status_reasons + ["route_not_transformed"])),
                    "style_class": properties.get("style_class", ""),
                }
            )
            continue

        final_review = needs_manual_review or used_page_fallback or transform_conf < 0.75 or model.quality_status != "pass"
        final_properties = dict(properties)
        final_properties.update(
            {
                "coordinate_space": "wgs84",
                "transform_scope": model.scope_id,
                "transform_model": model.model_name,
                "y_mode": model.y_mode,
                "model_quality_status": model.quality_status,
                "crs_candidate": model.crs_candidate,
                "gcp_count": model.gcp_count,
                "rmse_m": round(model.rmse_m, 3),
                "loocv_rmse_m": None if model.loocv_rmse_m is None else round(model.loocv_rmse_m, 3),
                "seam_err_m": None if model.seam_err_m is None else round(model.seam_err_m, 3),
                "map_matched": False,
                "confidence": round(transform_conf, 4),
                "needs_manual_review": final_review,
                "review_reasons": ",".join(dict.fromkeys(status_reasons)),
            }
        )
        transformed.append({"type": "Feature", "geometry": geometry, "properties": final_properties})
        route_status.append(
            {
                "route_id": route_id,
                "page_no": page_no,
                "frame_id": frame_id,
                "transform_scope": model.scope_id,
                "transform_model": model.model_name,
                "y_mode": model.y_mode,
                "model_quality_status": model.quality_status,
                "crs_candidate": model.crs_candidate,
                "gcp_count": model.gcp_count,
                "rmse_m": round(model.rmse_m, 3),
                "loocv_rmse_m": "" if model.loocv_rmse_m is None else round(model.loocv_rmse_m, 3),
                "seam_err_m": "" if model.seam_err_m is None else round(model.seam_err_m, 3),
                "confidence": round(transform_conf, 4),
                "needs_manual_review": final_review,
                "review_reasons": ",".join(dict.fromkeys(status_reasons)),
            }
        )

    return transformed, untransformed, route_status


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_frame_model_rows(models: dict[str, TransformModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted(models.values(), key=lambda item: (item.scope_type, item.page_no, item.frame_id or "")):
        rows.append(
            {
                "scope_id": model.scope_id,
                "scope_type": model.scope_type,
                "page_no": model.page_no,
                "frame_id": model.frame_id or "",
                "transform_model": model.model_name,
                "y_mode": model.y_mode,
                "crs_candidate": model.crs_candidate,
                "gcp_count": model.gcp_count,
                "manual_count": model.manual_count,
                "auto_count": model.auto_count,
                "gcp_mean_confidence": round(model.gcp_mean_confidence, 4),
                "rmse_m": round(model.rmse_m, 3),
                "loocv_rmse_m": "" if model.loocv_rmse_m is None else round(model.loocv_rmse_m, 3),
                "quality_status": model.quality_status,
                "seam_err_m": "" if model.seam_err_m is None else round(model.seam_err_m, 3),
                "duplicate_count": model.duplicate_count,
                "gcp_ids": "|".join(model.gcp_ids),
            }
        )
    return rows


def build_debug_html(token: str) -> str:
    config = json.dumps({"mapboxToken": token}, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Step 5 Route Debug</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.css" rel="stylesheet" />
  <style>
    :root {{
      --bg: #f2eee4;
      --panel: #fffaf1;
      --line: #d8cfbb;
      --ink: #201d18;
      --warn: #ba5b1d;
      --ok: #1b7a63;
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
      grid-template-columns: 340px 1fr;
      height: 100%;
    }}
    #sidebar {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    h1 {{
      font-size: 18px;
      margin: 0 0 12px;
    }}
    select, button {{
      width: 100%;
      padding: 8px 10px;
      margin-bottom: 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: white;
    }}
    .metric {{
      margin: 10px 0;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }}
    .metric strong {{
      display: block;
      font-size: 18px;
    }}
    .list {{
      font-size: 12px;
      line-height: 1.5;
      margin-top: 12px;
    }}
    .item {{
      margin-bottom: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .ok {{ color: var(--ok); font-weight: 700; }}
  </style>
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h1>Step 5 Route Debug</h1>
      <select id="pageSelect"></select>
      <button id="reviewToggle">レビュー対象のみ: OFF</button>
      <div class="metric"><strong id="routeCount">0</strong>transformed routes</div>
      <div class="metric"><strong id="missingCount">0</strong>untransformed routes</div>
      <div id="summary" class="list"></div>
      <div id="missing" class="list"></div>
    </aside>
    <main id="map"></main>
  </div>
  <script>window.DEBUG_CONFIG = {config};</script>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.5.1/mapbox-gl.js"></script>
  <script>
    const token = window.DEBUG_CONFIG.mapboxToken || '';
    if (token) mapboxgl.accessToken = token;
    const blankStyle = {{ version: 8, sources: {{}}, layers: [{{ id: 'bg', type: 'background', paint: {{ 'background-color': '#f6f2ea' }} }}] }};
    const state = {{ page: null, reviewOnly: false, showAll: false, routes: [], trusted: [], missing: [] }};
    const els = {{
      pageSelect: document.getElementById('pageSelect'),
      reviewToggle: document.getElementById('reviewToggle'),
      scopeToggle: null,
      routeCount: document.getElementById('routeCount'),
      missingCount: document.getElementById('missingCount'),
      summary: document.getElementById('summary'),
      missing: document.getElementById('missing'),
    }};
    const map = new mapboxgl.Map({{
      container: 'map',
      style: token ? 'mapbox://styles/mapbox/outdoors-v12' : blankStyle,
      center: [133.6, 33.8],
      zoom: 7.1
    }});
    async function loadData() {{
      const [routeRes, trustedRes, missingRes] = await Promise.all([
        fetch('transformed_routes.geojson'),
        fetch('trusted_routes.geojson'),
        fetch('untransformed_routes.json')
      ]);
      state.routes = (await routeRes.json()).features;
      state.trusted = (await trustedRes.json()).features;
      state.missing = await missingRes.json();
      const pages = Array.from(new Set([
        ...state.routes.map(f => f.properties.page_no),
        ...state.trusted.map(f => f.properties.page_no),
        ...state.missing.map(r => r.page_no),
      ])).sort((a, b) => a - b);
      state.page = pages[0] || null;
      els.pageSelect.innerHTML = pages.map(page => `<option value="${{page}}">Page ${{page}}</option>`).join('');
      const scopeToggle = document.createElement('button');
      scopeToggle.id = 'scopeToggle';
      scopeToggle.textContent = '表示: trusted only';
      els.pageSelect.insertAdjacentElement('afterend', scopeToggle);
      els.scopeToggle = scopeToggle;
      els.pageSelect.addEventListener('change', () => {{ state.page = Number(els.pageSelect.value); render(); }});
      els.reviewToggle.addEventListener('click', () => {{
        state.reviewOnly = !state.reviewOnly;
        els.reviewToggle.textContent = `レビュー対象のみ: ${{state.reviewOnly ? 'ON' : 'OFF'}}`;
        render();
      }});
      els.scopeToggle.addEventListener('click', () => {{
        state.showAll = !state.showAll;
        els.scopeToggle.textContent = `表示: ${{state.showAll ? 'all transformed' : 'trusted only'}}`;
        render();
      }});
      render();
    }}
    function currentRoutes() {{
      const source = state.showAll ? state.routes : state.trusted;
      return source.filter(route => {{
        if (route.properties.page_no !== state.page) return false;
        if (state.reviewOnly && !route.properties.needs_manual_review) return false;
        return true;
      }});
    }}
    function currentMissing() {{
      return state.missing.filter(route => route.page_no === state.page);
    }}
    function ensureLayers() {{
      if (!map.getSource('routes')) {{
        map.addSource('routes', {{ type: 'geojson', data: {{ type: 'FeatureCollection', features: [] }} }});
        map.addLayer({{
          id: 'routes-line',
          type: 'line',
          source: 'routes',
          paint: {{
            'line-color': [
              'case',
              ['boolean', ['get', 'needs_manual_review'], false], '#c55a11',
              '#ca2c2c'
            ],
            'line-width': [
              'interpolate', ['linear'], ['zoom'],
              7, 2,
              12, 5
            ],
            'line-dasharray': [
              'case',
              ['boolean', ['get', 'needs_manual_review'], false], ['literal', [1.5, 1.2]],
              ['literal', [1, 0]]
            ]
          }}
        }});
      }}
    }}
    function render() {{
      if (!map.isStyleLoaded()) return;
      ensureLayers();
      const routes = currentRoutes();
      const missing = currentMissing();
      map.getSource('routes').setData({{ type: 'FeatureCollection', features: routes }});
      els.routeCount.textContent = String(routes.length);
      els.missingCount.textContent = String(missing.length);
      els.summary.innerHTML = routes.map(route => {{
        const p = route.properties;
        return `<div class="item"><div class="${{p.needs_manual_review ? 'warn' : 'ok'}}">${{p.route_id}}</div><div>${{p.transform_model}} / ${{p.crs_candidate}}</div><div>gcp=${{p.gcp_count}} rmse=${{p.rmse_m}} loocv=${{p.loocv_rmse_m ?? '-'}} seam=${{p.seam_err_m ?? '-'}} confidence=${{p.confidence}}</div><div>${{p.review_reasons || 'trusted route'}}</div></div>`;
      }}).join('') || '<div class="item">該当ルートなし</div>';
      els.missing.innerHTML = missing.map(route => `<div class="item"><div class="warn">${{route.route_id}}</div><div>${{route.review_reasons}}</div></div>`).join('');
      if (routes.length) {{
        const bounds = new mapboxgl.LngLatBounds();
        const addCoord = coord => bounds.extend(coord);
        for (const route of routes) {{
          if (route.geometry.type === 'LineString') {{
            route.geometry.coordinates.forEach(addCoord);
          }} else {{
            route.geometry.coordinates.forEach(line => line.forEach(addCoord));
          }}
        }}
        map.fitBounds(bounds, {{ padding: 48, maxZoom: 13 }});
      }}
    }}
    map.on('load', render);
    loadData();
  </script>
</body>
</html>
"""


def write_report(
    path: Path,
    transformed_count: int,
    untransformed_count: int,
    route_status: list[dict[str, Any]],
    frame_models: list[dict[str, Any]],
) -> None:
    model_counts = Counter(row["transform_model"] for row in frame_models)
    scope_counts = Counter(row["scope_type"] for row in frame_models)
    status_counts = Counter()
    for row in route_status:
        if row["transform_model"]:
            status_counts["transformed"] += 1
        else:
            status_counts["untransformed"] += 1
    lines = [
        "# Step 5 Report",
        "",
        f"- Transformed routes: `{transformed_count}`",
        f"- Untransformed routes: `{untransformed_count}`",
        f"- Selected transform scopes: `{len(frame_models)}`",
        "",
        "## Scope Counts",
    ]
    for key, value in sorted(scope_counts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Model Counts"])
    for key, value in sorted(model_counts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "- Frame/page models: `artifacts/step5/frame_models.csv`",
            "- Candidate diagnostics: `artifacts/step5/model_candidates.csv`",
            "- Route transform status: `artifacts/step5/route_transform_status.csv`",
            "- Transformed routes: `artifacts/step5/transformed_routes.geojson`",
            "- Trusted routes only: `artifacts/step5/trusted_routes.geojson`",
            "- Mapbox debug HTML: `artifacts/step5/mapbox_route_debug/index.html`",
            "",
            "## Notes",
            "- 標準の優先順位は `y_flipped + similarity` です。高自由度モデルは LOOCV が明確に改善した場合だけ採用します。",
            "- `page_model_fallback` は debug 用の `--allow-page-fallback` を有効にした時だけ使います。本番既定では禁止です。",
            "- `missing_frame_model` / `missing_page_model` は Step 4 の GCP 密度不足、または採用条件を満たさず未変換になったルートです。",
            "- `seam_err_m` は重複寺院ラベルから計算したページ間継ぎ目の proxy です。Step 6 で本格的な seam 最適化を入れます。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: georeference route geometries from GCP candidates")
    parser.add_argument("--routes", type=Path, default=Path("artifacts/step3/merged_routes.geojson"))
    parser.add_argument("--gcps", type=Path, default=Path("artifacts/step4/gcp_candidates.csv"))
    parser.add_argument("--page-metadata", type=Path, default=Path("artifacts/step1/page_red_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/step5"))
    parser.add_argument("--allow-page-fallback", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    page_heights = load_page_heights(args.page_metadata)
    gcps = load_gcps(args.gcps, page_heights)
    routes = load_routes(args.routes)
    models, candidate_rows = select_best_models(gcps)
    compute_seam_errors(models, gcps)
    transformed, untransformed, route_status = build_outputs(
        routes,
        models,
        page_heights,
        allow_page_fallback=args.allow_page_fallback,
    )
    frame_model_rows = build_frame_model_rows(models)
    trusted_routes = [feature for feature in transformed if not feature["properties"]["needs_manual_review"]]

    write_csv(args.out_dir / "frame_models.csv", frame_model_rows)
    write_csv(args.out_dir / "model_candidates.csv", candidate_rows)
    write_csv(args.out_dir / "route_transform_status.csv", route_status)
    write_geojson(args.out_dir / "transformed_routes.geojson", transformed)
    write_geojson(args.out_dir / "trusted_routes.geojson", trusted_routes)
    (args.out_dir / "untransformed_routes.json").write_text(json.dumps(untransformed, ensure_ascii=False, indent=2), encoding="utf-8")
    extraction_log = {
        "gcp_count": len(gcps),
        "selected_model_count": len(models),
        "transformed_route_count": len(transformed),
        "untransformed_route_count": len(untransformed),
    }
    (args.out_dir / "extraction_log.json").write_text(json.dumps(extraction_log, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.out_dir / "step5_report.md", len(transformed), len(untransformed), route_status, frame_model_rows)

    debug_dir = args.out_dir / "mapbox_route_debug"
    ensure_dir(debug_dir)
    (debug_dir / "transformed_routes.geojson").write_text((args.out_dir / "transformed_routes.geojson").read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "trusted_routes.geojson").write_text((args.out_dir / "trusted_routes.geojson").read_text(encoding="utf-8"), encoding="utf-8")
    (debug_dir / "untransformed_routes.json").write_text(json.dumps(untransformed, ensure_ascii=False, indent=2), encoding="utf-8")
    (debug_dir / "index.html").write_text(build_debug_html(read_mapbox_token()), encoding="utf-8")


if __name__ == "__main__":
    main()
