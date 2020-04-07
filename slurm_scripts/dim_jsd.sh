#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh p100  local_infomax_encoder_jsd_2_full 2 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator jsd --gpu \
                  --prefix local_infomax_encoder_jsd_2_full --epoch 300 --batch_size 64"
