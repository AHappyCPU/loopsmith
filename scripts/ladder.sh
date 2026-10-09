#!/bin/bash
# Trains the submission on a ladder of gen_quad_ladder.py datasets, from one
# modulus up to 2000, with a 15-minute budget per stage, and prints the T=1 and
# T=2 results for each.
source "$(dirname "$0")/env.sh"

run_stage () {
  name=$1; shift
  echo "=== STAGE $name ==="
  "$PYTHON" "$REPO/scripts/gen_quad_ladder.py" --out data/generated/quad_$name "$@" || return 1
  sed "s#h1_replica_v1#quad_$name#; s#h1-replica#quad-$name#; s#\"total_training_time_seconds\": 7200#\"total_training_time_seconds\": 900#" \
    "$REPO/manifests/h1_replica.json" > benchmark/manifests/quad_$name.json
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  LOOPSMITH_BS="${LOOPSMITH_BS:-256}" LOOPSMITH_EVAL_BS="${LOOPSMITH_EVAL_BS:-512}" \
  LOOPSMITH_DIM="${LOOPSMITH_DIM:-256}" \
    "$PYTHON" -m benchmark.runner \
    --manifest benchmark/manifests/quad_$name.json \
    --submission-file "$REPO/submission.py" 2>&1 \
    | grep -E "depth_t=1 |depth_t=2 |RESULT_JSON" | head -4
  echo "=== END $name ==="
}
run_stage w1_1mod   --moduli 1    --c_mode full --t_set 1 --rows 6000
run_stage w2_32mod  --moduli 32   --c_mode full --t_set 1 --rows 100000
run_stage w3_512mod --moduli 512  --c_mode full --t_set 1 --rows 300000
run_stage w4_2kmod  --moduli 2000 --c_mode full --t_set 1 --rows 500000
run_stage w5_2k_c0  --moduli 2000 --c_mode zero --t_set 1 --rows 500000
echo "LADDER COMPLETE"
