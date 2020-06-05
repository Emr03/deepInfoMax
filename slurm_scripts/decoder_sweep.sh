#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/local_infomax_encoder_*_mnist; do
  name=${dir/experiments\/}
  encoder_ckpt=${dir}/${name}_checkpoint.pth
  echo ${dir}
  # partition, j_name, resource cmd
  bash slurm_scripts/launch_slurm_job.sh p100 decoder_${name} 1 \
  "python3 decoding.py --encoder_ckpt=${encoder_ckpt} --gpu \
  --prefix decoder_${name} --data=mnist --code_size=8 \
		  --epochs=1000 --batch_size=256"
  done


