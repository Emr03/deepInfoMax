#!/bin/bash

# set up environment
# load the conda profile
. /h/elsinator/miniconda3/etc/profile.d/conda.sh

# activate conda env
conda activate infomax

# symlink checkpoint directory to run directory
# ln -s /checkpoint/$USER/$SLURM_JOB_ID /h/elsinator/deepInfoMax/experiments/$SLURM_JOB_ID

python3 deep_infomax.py --encoder_stride 2 --mi_estimator jsd --gpu --prefix local_infomax_encoder_jsd_2
