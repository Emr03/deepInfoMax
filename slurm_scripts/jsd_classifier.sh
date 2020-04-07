#!/bin/bash


for mode in "fc" "conv" "y"; do
  python3 classification.py --encoder_stride 2 --encoder_ckpt=experiments/local_infomax_encoder_jsd_2 --input_layer ${mode} --gpu \
                    --prefix clasksifier_jsd_2_${mode}
done