#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/encoders/prior/local_infomax_encoder_*_no_sigmoid; do
  name=${dir/experiments\/encoders\/prior\/}
  encoder_ckpt=${dir}/${name}_checkpoint.pth
  echo ${dir}
  # partition, j_name, resource cmd
  bash slurm_scripts/launch_slurm_job.sh p100 decoder_${name} 1 \
  "python3 decoding.py --encoder_ckpt=${encoder_ckpt} --gpu \
  --prefix decoder_${name}_new --decoder_ckpt=experiments/decoder_${name}_new/decoder_${name}_new_checkpoint.pth \
		  --epochs=7000 --batch_size=256"
  done


