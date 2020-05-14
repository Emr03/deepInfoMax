#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/encoders/prior/local_infomax_encoder_*_no_sigmoid; do
  name=${dir/experiments\/encoders\/prior\/}
  encoder_ckpt=${dir}/${name}_checkpoint.pth
  echo ${dir}
  # partition, j_name, resource cmd
  bash slurm_scripts/launch_slurm_job.sh gpu decoder_${name} 1 \
  "python3 decoding.py --encoder_ckpt=${encoder_ckpt} --gpu \
                  --prefix decoder_${name} --epochs=3000"
  done


