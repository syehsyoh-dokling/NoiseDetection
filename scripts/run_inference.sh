#!/usr/bin/env bash
# Track 1 inference: test WAVs -> predictions.jsonl in a zip (Phase 2 format).
#   bash scripts/run_inference.sh /path/to/test_wavs [out.zip]
# Prerequisites (README): data/validation/ (official release, for the template library and null levels),
# the release v1.0 checkpoints and BEATs_iter3_plus_AS2M.pt in models/.
set -euo pipefail
cd "$(dirname "$0")/.."
TEST_DIR=${1:?usage: run_inference.sh /path/to/test_wavs [out.zip]}
OUT=${2:-work/submission_track1.zip}
if [ ! -f cache/template_library.pkl ] || [ ! -f cache/val_template_cands.pkl ]; then
  python eval_templates.py          # library from validation (noisy - clean) + validation candidates for the null levels
fi
python inference/infer.py --test-dir "$TEST_DIR" --out "$OUT"
