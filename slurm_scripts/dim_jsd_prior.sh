#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh gpu local_infomax_encoder_jsd_prior_matching_new 1 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator js --gpu \
                  --prefix local_infomax_encoder_jsd_prior_matching_new --prior_matching --epoch 1000 --batch_size 64"
