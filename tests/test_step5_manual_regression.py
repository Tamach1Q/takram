import json
import unittest
from pathlib import Path

from scripts.georeference_routes import GCP, fit_candidate_model, select_best_models


class Step5ManualRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        data = json.loads(Path("data/manual_gcps/page_061_061_f06.json").read_text(encoding="utf-8"))
        page_height = float(data["page"]["rect_pt"]["y1"]) - float(data["page"]["rect_pt"]["y0"])
        self.gcps: list[GCP] = []
        for index, row in enumerate(data["gcps"], start=1):
            if row.get("role") != "frame_anchor":
                continue
            self.gcps.append(
                GCP(
                    gcp_id=f"manual_{index:02d}",
                    page_no=int(data["page_no"]),
                    frame_id=str(data["frame_id"]),
                    temple_group="manual",
                    temple_no=index,
                    pdf_x=float(row["pdf_x"]),
                    pdf_y=float(row["raw_pdf_y_top_left"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    confidence=1.0,
                    needs_manual_review=False,
                    page_height_pt=page_height,
                )
            )

    def test_y_flipped_similarity_loocv_regression(self) -> None:
        model = fit_candidate_model(
            self.gcps,
            scope_type="frame",
            scope_id="frame:61:061_f06",
            page_no=61,
            frame_id="061_f06",
            model_name="similarity",
            y_mode="y_flipped",
            crs_candidate="local_equirect_m",
        )
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.model_name, "similarity")
        self.assertEqual(model.y_mode, "y_flipped")
        self.assertIsNotNone(model.loocv_rmse_m)
        self.assertAlmostEqual(model.loocv_rmse_m or 0.0, 35.97, delta=1.0)
        self.assertEqual(model.quality_status, "pass")

    def test_selector_prefers_y_flipped_similarity(self) -> None:
        selected, _ = select_best_models(self.gcps)
        model = selected["frame:61:061_f06"]
        self.assertEqual(model.model_name, "similarity")
        self.assertEqual(model.y_mode, "y_flipped")
        self.assertEqual(model.crs_candidate, "local_equirect_m")


if __name__ == "__main__":
    unittest.main()
