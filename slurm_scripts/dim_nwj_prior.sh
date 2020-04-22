#!/bin/bash

bash slurm_scripts/launch_slurm_job.sh gpu local_infomax_encoder_nwj_new_prior 1 \
    "python3 deep_infomax.py --encoder_stride 2 --mi_estimator nwj --gpu \
                  --prefix local_infomax_encoder_nwJ_new_prior --prior_matching --epoch 1000 --batch_size 64"
