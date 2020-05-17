#!/bin/bash

cd ~/deepInfoMax
mode=fc
stride=2
for mi_estimator in nce dv js; do
 dir=experiments/infomax_classifiers/classifier_${mode}_local_infomax_encoder_${mi_estimator}_no_sigmoid
 name=${dir/experiments\/infomax_classifiers\/}
 classifier_ckpt=${dir}/${name}_checkpoint.pth
 encoder_name=${name/classifier_${mode}_}
 encoder_ckpt=experiments/encoders/prior/${encoder_name}/${encoder_name}_checkpoint.pth
 python3 attack_infomax.py --classifier_ckpt=${classifier_ckpt} --encoder_ckpt=${encoder_ckpt} --input_layer ${mode} \
    --gpu --mi_estimator ${mi_estimator} --prefix ${dir/experiments}  --batch_size=64
done
