#!/bin/bash

# loop over encoder_ckpts
for dir in experiments/local_infomax_encoder_*; do
  echo ${dir}
  dir=${dir/experiments\//}
  echo ${dir}
  encoder_ckpt=experiments/${dir}/${dir}_checkpoint.pth
  # partition, j_name, resource cmd
  bash slurm_scripts/launch_slurm_job.sh p100 ndm_${dir} 1 \
  "python3 ndm_eval.py --encoder_stride ${stride} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} --gpu \
                --prefix ${dir} --batch_size=64 --lr=0.0001 --epochs=500"
done