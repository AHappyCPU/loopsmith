#!/bin/bash
# Generates an m1-style dataset with the competition's own generator, then trains
# and evaluates loopsmith_j on it under the H1 replica runtime settings.
source "$(dirname "$0")/env.sh"

"$PYTHON" -m data.squaring_mod --output_dir data/generated/m1_local \
  --fixed_p 101 --fixed_q 103 --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 10000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 192 \
  --ood_n_depth_evaluation_modulus_bits '[15,16]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 \
  --separate_input_output true
sed 's#h1_replica_v1#m1_local#; s#h1-replica#m1-local-j#' \
  "$REPO/manifests/h1_replica.json" > benchmark/manifests/m1_j.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LOOPSMITH_BS="${LOOPSMITH_BS:-320}" LOOPSMITH_EVAL_BS="${LOOPSMITH_EVAL_BS:-512}" \
  "$PYTHON" -m benchmark.runner \
  --manifest benchmark/manifests/m1_j.json \
  --submission-file "$REPO/experiments/loopsmith_j/submission.py"
