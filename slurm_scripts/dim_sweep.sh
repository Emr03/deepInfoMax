#!/bin/bash

for est in "jsd" "dv" "nce"; do
  for stride in 1 2; do
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh p100  local_infomax_encoder_${est}_${stride}_full 4 \
    "python3 deep_infomax.py --encoder_stride ${stride} --mi_estimator ${est} --gpu \
                  --prefix local_infomax_encoder_${est}_${stride}"
  done
done
