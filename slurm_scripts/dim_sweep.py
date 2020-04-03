import os
estimator = ["JSD", "DV", "NCE"]
encoder_stride = [1, 2]

for est in estimator:
    for stride in encoder_stride:
        os.system("launch_slurm_job.sh gpu local_infomax_encoder_${est}_${stride} 1 "
                  "python3 deep_infomax.py --encoder_stride ${stride} --mi_estimator ${est} --gpu "
                  "--prefix local_infomax_encoder_${est}_${stride}")

        os.system("launch_slurm_job.sh gpu global_infomax_encoder_${est}_${stride} 1 "
                  "python3 deep_infomax.py --encoder_stride ${stride} --mi_estimator ${est} --gpu "
                  "--prefix global_infomax_encoder_${est}_${stride} --global_dim")

