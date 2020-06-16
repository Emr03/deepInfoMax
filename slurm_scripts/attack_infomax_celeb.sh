#!/bin/bash

for mi_estimator in js dv nce; do
 dir=experiments/encoders/celeb
 encoder_name=local_infomax_encoder_${mi_estimator}_celeb
 encoder_ckpt=dir/${encoder_name}/${encoder_name}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh p100 attack_infomax_${mi_estimator} 1 \
    "python3 attack_infomax.py --encoder_ckpt=${encoder_ckpt} \
    --gpu --mi_estimator ${mi_estimator} --prefix ${dir/experiments}/${encoder_name}  --batch_size=64"
done