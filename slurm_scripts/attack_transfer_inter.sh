#!/bin/bash

for estimator1 in js dv nce; do
  dir1=experiments/encoders/prior/local_infomax_encoder_${estimator1}_no_sigmoid
  encoder_name1=${dir1/experiments\/encoders\/prior\/}
  encoder_ckpt1=experiments/encoders/prior/${encoder_name1}/${encoder_name1}_checkpoint.pth
  for estimator2 in js dv nce; do
    if [[ ${estimator1} == ${estimator2} ]]; then
      continue
    fi
    dir2=experiments/encoders/prior/local_infomax_encoder_${estimator2}_no_sigmoid
    encoder_name2=${dir2/experiments\/encoders\/prior\/}
    encoder_ckpt2=experiments/encoders/prior/${encoder_name2}/${encoder_name2}_checkpoint.pth
    python3 attack_transfer.py --source_model_ckpt=${encoder_ckpt1} --target_model_ckpt=${encoder_ckpt2} --gpu \
    --prefix=encoder_attack_transfer --log=${estimator1}_prior_to_${estimator2}_prior.log 
  done
done
