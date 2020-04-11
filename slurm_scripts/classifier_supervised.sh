#!/bin/bash


bash slurm_scripts/launch_slurm_job.sh gpu  classifier_supervised 1 \
    "python3 classification.py --encoder_stride 2 --fully_supervised  --gpu \
                  --prefix classifier_supervised"