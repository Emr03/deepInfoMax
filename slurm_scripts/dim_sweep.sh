#!/bin/bash

for est in "dv" "nce"; do
    # partition, j_name, resource cmd
    bash slurm_scripts/launch_slurm_job.sh gpu local_infomax_encoder_${est} 1 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator ${est} --prior_matching --gpu \
                  --prefix local_infomax_encoder_${est}_1 --gamma=1 --epochs=1000 --batch_size=64"
  done
