#!/bin/bash

stride=2
encoder_ckpt=experiments/local_infomax_encoder_jsd_new_prior/local_infomax_encoder_jsd_new_prior_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu decoder_jsd_05 1 \
  "python3 decoding.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --gpu \
                  --prefix decoder_jsd_05 --batch_size=128 --epochs=2000"
