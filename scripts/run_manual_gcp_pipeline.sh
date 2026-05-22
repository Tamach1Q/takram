#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/merge_manual_gcps.py
.venv/bin/python scripts/georeference_routes.py --gcps artifacts/step5/gcp_candidates_merged.csv
.venv/bin/python scripts/build_manual_review_queue.py
