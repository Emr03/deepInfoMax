#!/bin/bash
stride=2

for dir in experiments/supervised/classifier_supervised_*; do
 prefix=${dir/experiments\/}
 name=${prefix/supervised\/}
 mode=${name/classifier_supervised_}
 ckpt=${dir}/${name}_checkpoint.pth
 bash slurm_scripts/launch_slurm_job.sh gpu classifier_stats_${name} 1 \
        "python stability_test.py --encoder_stride ${stride} --classifier_ckpt=$ckpt \
        --input_layer ${mode} --gpu --num_steps=100 --epsilon=0.05 --prefix=${prefix} --hidden_units=200 --batch_size=64"
done

