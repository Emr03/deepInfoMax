#!/bin/bash

mode=fc
mi_estimator=nce
stride=2
for dir in experiments/infomax_classifiers/classifier_${mode}_local_infomax_encoder_${mi_estimator}_*; do
name=${dir\/experiments\/infomax_classifiers\/}
classifier_ckpt=${dir}/${name}_checkpoint.pth
encoder_name=${name\/_${mode}_}
encoder_ckpt=experiments/encoders/prior/${encoder_name}/${encoder_name}_checkpoint.pth
bash slurm_scripts/launch_slurm_job.sh gpu attack_infomax 1 \
    "python3 attack_infomax.py --classifier_ckpt=${classifier_ckpt} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} \
    --gpu --mi_estimator ${mi_estimator} --prefix ${dir\/experiments}  --batch_size=64"
