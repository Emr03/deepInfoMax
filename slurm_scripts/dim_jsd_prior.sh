#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh gpu local_infomax_encoder_jsd_new_prior 2 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator js --gpu \
                  --prefix local_infomax_encoder_jsd_new_prior --prior_matching --epoch 1000 --batch_size 64"