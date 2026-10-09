#!/bin/bash
# Generates an H1 replica dataset with gen_h1_replica_v4.py, then trains and
# evaluates loopsmith_j on it.
source "$(dirname "$0")/env.sh"

"$PYTHON" "$REPO/scripts/gen_h1_replica_v4.py" --out data/generated/h1_replica_v4c --seed 19
sed 's#h1_replica_v1#h1_replica_v4c#; s#h1-replica#h1-replica-v4c#' \
  "$REPO/manifests/h1_replica.json" > benchmark/manifests/h1_v4c.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LOOPSMITH_BS="${LOOPSMITH_BS:-320}" LOOPSMITH_EVAL_BS="${LOOPSMITH_EVAL_BS:-512}" \
  "$PYTHON" -m benchmark.runner \
  --manifest benchmark/manifests/h1_v4c.json \
  --submission-file "$REPO/experiments/loopsmith_j/submission.py"
