#!/bin/bash

mode=fc
stride=2
for mi_estimator in js dv nce; do
 dir=experiments/infomax_classifiers/classifier_${mode}_local_infomax_encoder_${mi_estimator}_no_sigmoid
 name=${dir/experiments\/infomax_classifiers\/}
 classifier_ckpt=${dir}/${name}_checkpoint.pth
 encoder_name=${name/classifier_${mode}_}
 encoder_ckpt=experiments/encoders/prior/${encoder_name}/${encoder_name}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh p100 attack_infomax_${mi_estimator} 1 \
    "python3 attack_infomax.py --classifier_ckpt=${classifier_ckpt} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} \
    --gpu --mi_estimator ${mi_estimator} --prefix ${dir/experiments}  --batch_size=64"
done
