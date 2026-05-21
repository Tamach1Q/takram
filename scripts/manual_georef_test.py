#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EARTH_RADIUS_M = 6378137.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lonlat_to_local_equirect(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = EARTH_RADIUS_M * math.radians(lat - ref_lat)
    return (x, y)


def local_equirect_to_lonlat(x: float, y: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    return (lon, lat)


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


TRANSFORMS = {
    "similarity": (2, fit_similarity, apply_similarity),
    "affine": (3, fit_affine, apply_affine),
}


def load_manual_gcps(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def valid_anchor_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in data.get("gcps", []):
        if row.get("role") != "frame_anchor":
            continue
        if any(key not in row for key in ["pdf_x", "raw_pdf_y_top_left", "longitude", "latitude"]):
            continue
        rows.append(row)
    return rows


def y_modes(row: dict[str, Any], page_height: float) -> dict[str, float]:
    raw_top = float(row["raw_pdf_y_top_left"])
    return {
      "raw_y": raw_top,
      "y_flipped": page_height - raw_top,
    }


def rms(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(math.sqrt(sum(value * value for value in values) / len(values)))


def residual_rows_for_combo(
    anchors: list[dict[str, Any]],
    page_height: float,
    y_mode: str,
    transform_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_points, fit_fn, apply_fn = TRANSFORMS[transform_name]
    if len(anchors) < min_points:
        raise ValueError(f"{transform_name} requires at least {min_points} points")

    ref_lon = sum(float(row["longitude"]) for row in anchors) / len(anchors)
    ref_lat = sum(float(row["latitude"]) for row in anchors) / len(anchors)

    source = np.array(
        [(float(row["pdf_x"]), y_modes(row, page_height)[y_mode]) for row in anchors],
        dtype=float,
    )
    target = np.array(
        [lonlat_to_local_equirect(float(row["longitude"]), float(row["latitude"]), ref_lon, ref_lat) for row in anchors],
        dtype=float,
    )

    params = fit_fn(source, target)
    predicted = apply_fn(source, params)

    residual_rows: list[dict[str, Any]] = []
    fit_errors: list[float] = []
    loocv_errors: list[float] = []
    for index, row in enumerate(anchors):
        dx = float(predicted[index][0] - target[index][0])
        dy = float(predicted[index][1] - target[index][1])
        fit_error = float(math.hypot(dx, dy))
        fit_errors.append(fit_error)
        pred_lon, pred_lat = local_equirect_to_lonlat(predicted[index][0], predicted[index][1], ref_lon, ref_lat)

        loocv_dx = loocv_dy = loocv_error = None
        loocv_lon = loocv_lat = None
        if len(anchors) - 1 >= min_points:
            train_idx = [i for i in range(len(anchors)) if i != index]
            train_source = source[train_idx]
            train_target = target[train_idx]
            train_params = fit_fn(train_source, train_target)
            loocv_pred = apply_fn(source[index:index + 1], train_params)[0]
            loocv_dx = float(loocv_pred[0] - target[index][0])
            loocv_dy = float(loocv_pred[1] - target[index][1])
            loocv_error = float(math.hypot(loocv_dx, loocv_dy))
            loocv_lon, loocv_lat = local_equirect_to_lonlat(loocv_pred[0], loocv_pred[1], ref_lon, ref_lat)
            loocv_errors.append(loocv_error)

        residual_rows.append(
            {
                "combo": f"{y_mode}__{transform_name}",
                "index": row.get("index", index + 1),
                "name": row.get("name", ""),
                "role": row.get("role", ""),
                "pdf_x": round(float(row["pdf_x"]), 3),
                "pdf_y": round(y_modes(row, page_height)[y_mode], 3),
                "target_longitude": round(float(row["longitude"]), 7),
                "target_latitude": round(float(row["latitude"]), 7),
                "fit_pred_longitude": round(pred_lon, 7),
                "fit_pred_latitude": round(pred_lat, 7),
                "fit_residual_dx_m": round(dx, 3),
                "fit_residual_dy_m": round(dy, 3),
                "fit_residual_m": round(fit_error, 3),
                "loocv_pred_longitude": "" if loocv_lon is None else round(loocv_lon, 7),
                "loocv_pred_latitude": "" if loocv_lat is None else round(loocv_lat, 7),
                "loocv_residual_dx_m": "" if loocv_dx is None else round(loocv_dx, 3),
                "loocv_residual_dy_m": "" if loocv_dy is None else round(loocv_dy, 3),
                "loocv_residual_m": "" if loocv_error is None else round(loocv_error, 3),
            }
        )

    summary = {
        "combo": f"{y_mode}__{transform_name}",
        "page_no": data_page_no(anchors),
        "frame_id": data_frame_id(anchors),
        "gcp_count": len(anchors),
        "fit_rmse_m": round(rms(fit_errors), 3),
        "fit_max_m": round(max(fit_errors), 3),
        "loocv_rmse_m": "" if not loocv_errors else round(rms(loocv_errors), 3),
        "loocv_max_m": "" if not loocv_errors else round(max(loocv_errors), 3),
    }
    return residual_rows, summary


def data_page_no(anchors: list[dict[str, Any]]) -> int | str:
    first = anchors[0] if anchors else {}
    return first.get("page_no", "")


def data_frame_id(anchors: list[dict[str, Any]]) -> str:
    first = anchors[0] if anchors else {}
    return first.get("frame_id", "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_geojson(path: Path, residual_rows: list[dict[str, Any]]) -> None:
    features: list[dict[str, Any]] = []
    for row in residual_rows:
        if row["fit_pred_longitude"] == "":
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(row["target_longitude"]), float(row["target_latitude"])],
                        [float(row["fit_pred_longitude"]), float(row["fit_pred_latitude"])],
                    ],
                },
                "properties": {
                    "combo": row["combo"],
                    "index": row["index"],
                    "name": row["name"],
                    "fit_residual_m": row["fit_residual_m"],
                    "loocv_residual_m": row["loocv_residual_m"],
                },
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, input_path: Path, comparison_rows: list[dict[str, Any]], anchor_count: int) -> None:
    lines = [
        "# Manual Georef Test",
        "",
        f"- Input JSON: `{input_path}`",
        f"- Frame anchors used: `{anchor_count}`",
        "",
        "## Combo Summary",
    ]
    for row in comparison_rows:
        lines.append(
            f"- `{row['combo']}`: fit RMSE `{row['fit_rmse_m']}` m, "
            f"LOOCV RMSE `{row['loocv_rmse_m']}` m, LOOCV max `{row['loocv_max_m']}` m"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `comparison.csv`",
            "- `residual_vectors.csv`",
            "- `residual_vectors.geojson`",
            "",
            "## Notes",
            "- `frame_anchor` だけを使用しています。`route_trace` は無視します。",
            "- 比較は `raw_y / y_flipped × similarity / affine` の4通りです。",
            "- 赤線の変換前に、まず GCP 自体の残差と LOOCV を見る前提です。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare manual frame-anchor GCP transforms before route georeferencing")
    parser.add_argument(
        "input_json",
        type=Path,
        help="manual GCP JSON exported from artifacts/gcp_picker/index.html",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/manual_georef_test"),
    )
    args = parser.parse_args()

    data = load_manual_gcps(args.input_json)
    anchors = valid_anchor_rows(data)
    if len(anchors) < 3:
        raise SystemExit("frame_anchor が3点未満です。少なくとも3点、できれば5点以上を入れてください。")

    page_height = float(data["page"]["rect_pt"]["y1"]) - float(data["page"]["rect_pt"]["y0"])
    for index, row in enumerate(anchors, start=1):
        row["index"] = row.get("index", index)
        row["page_no"] = data.get("page_no", "")
        row["frame_id"] = data.get("frame_id", "")

    out_dir = args.out_dir / f"page_{int(data['page_no']):03d}_{data['frame_id']}"
    ensure_dir(out_dir)

    all_residuals: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for y_mode in ["raw_y", "y_flipped"]:
        for transform_name in ["similarity", "affine"]:
            residuals, summary = residual_rows_for_combo(
                anchors=anchors,
                page_height=page_height,
                y_mode=y_mode,
                transform_name=transform_name,
            )
            all_residuals.extend(residuals)
            comparison_rows.append(summary)

    write_csv(out_dir / "comparison.csv", comparison_rows)
    write_csv(out_dir / "residual_vectors.csv", all_residuals)
    write_geojson(out_dir / "residual_vectors.geojson", all_residuals)
    write_report(out_dir / "report.md", args.input_json, comparison_rows, len(anchors))


if __name__ == "__main__":
    main()
