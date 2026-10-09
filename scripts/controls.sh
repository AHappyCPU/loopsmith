#!/bin/bash
# Three control runs of the submission on single-modulus gen_quad_ladder.py
# datasets, 15 minutes each, printing the last few training and depth lines.
source "$(dirname "$0")/env.sh"

run_ctl () {
  name=$1; dim=$2; shift 2
  echo "=== CTL $name (dim $dim) ==="
  "$PYTHON" "$REPO/scripts/gen_quad_ladder.py" --out data/generated/ctl_$name "$@"
  sed "s#h1_replica_v1#ctl_$name#; s#h1-replica#ctl-$name#; s#\"total_training_time_seconds\": 7200#\"total_training_time_seconds\": 900#" \
    "$REPO/manifests/h1_replica.json" > benchmark/manifests/ctl_$name.json
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  LOOPSMITH_BS="${LOOPSMITH_BS:-256}" LOOPSMITH_EVAL_BS="${LOOPSMITH_EVAL_BS:-512}" \
  LOOPSMITH_DIM=$dim \
    "$PYTHON" -m benchmark.runner \
    --manifest benchmark/manifests/ctl_$name.json \
    --submission-file "$REPO/submission.py" 2>&1 \
    | grep -E "step=|depth_t=|RESULT_JSON" | tail -6
  echo "=== ENDCTL $name ==="
}
run_ctl a_c0     256 --moduli 1 --c_mode zero --t_set 1 --rows 6000 --bits 13
run_ctl b_d512   512 --moduli 1 --c_mode full --t_set 1 --rows 6000 --bits 13
run_ctl c_smalln 256 --moduli 1 --c_mode full --t_set 1 --rows 1200 --bits 11
echo "CONTROLS COMPLETE"
