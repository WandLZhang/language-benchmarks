#!/usr/bin/env bash
# Add ONE new model to an already-judged task without re-generating the whole field.
#
# Comparative judging groups candidates by (item, condition) and scores them against each other,
# so a new model means the whole group is re-judged — but only the new model is re-generated.
#   generate new model -> merge into the prior run -> re-judge all -> report -> leaderboard
#
# Usage: bash scripts/add_model.sh <task-dir> <model-id> <prior-results-dir> <out-dir> [testset] [workers]
# e.g.   bash scripts/add_model.sh tasks/cantonese-colloquial-translation claude-opus-5 \
#            tasks/cantonese-colloquial-translation/results/FINAL2 \
#            tasks/cantonese-colloquial-translation/results/FINAL3 testset_full_local.jsonl 5
set -euo pipefail

TASK="${1:?task dir}"; MODEL="${2:?model id}"; PRIOR="${3:?prior results dir}"; OUT="${4:?out dir}"
TESTSET="${5:-}"; WORKERS="${6:-5}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

TS_ARG=(); [ -n "$TESTSET" ] && TS_ARG=(--testset "$TESTSET")

echo "### 1/5 generate: $MODEL (contexts none,web)"
python engine/run_bench.py "$TASK" --contexts none,web --models "$MODEL" \
  "${TS_ARG[@]}" --workers "$WORKERS"
NEW=$(ls -td "$TASK"/results/*/ | head -1)
echo "new run: $NEW"

echo "### 2/5 merge with prior field"
python engine/merge_runs.py "$OUT" "$PRIOR" "$NEW"

echo "### 3/5 judge (comparative, whole field re-scored)"
python engine/judge.py "$TASK" "$OUT"

echo "### 4/5 report"
python engine/report.py "$TASK" "$OUT"

echo "### 5/5 leaderboard -> README.md"
python engine/leaderboard.py

echo
echo "DONE. Results in $OUT ; README.md regenerated."
