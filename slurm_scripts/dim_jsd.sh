#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh p100 local_infomax_encoder_jsd_no_prior 1 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator js --gpu \
                  --prefix local_infomax_encoder_jsd_no_prior --epoch 1000 --batch_size 64"
