#!/bin/bash
# loop over encoder_ckpts
for dir in experiments/encoders/prior/local_infomax_encoder_*; do
  echo ${dir}
  dir=${dir/experiments\/encoders\/prior\//}
  echo ${dir}
  encoder_ckpt=experiments/encoders/prior/${dir}/${dir}_checkpoint.pth
  echo ${encoder_ckpt}
  # partition, j_name, resource cmd
  bash slurm_scripts/launch_slurm_job.sh gpu ndm_${dir} 1 \
  "CUDA_LAUNCH_BLOCKING=1 python3 ndm_eval.py --encoder_stride 2 --encoder_ckpt=${encoder_ckpt} --gpu \
                --prefix ${dir} --batch_size=64 --lr=0.0001 --epochs=500"
done
