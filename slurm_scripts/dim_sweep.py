import os
estimator = ["jsd", "dv", "nce"]
encoder_stride = [1, 2]

for est in estimator:
    for stride in encoder_stride:
        os.system("bash slurm_scripts/launch_slurm_job.sh p100  local_infomax_encoder_{est}_{stride} 1 "
                  "python3 deep_infomax.py --encoder_stride {stride} --mi_estimator {est} --gpu "
                  "--prefix local_infomax_encoder_{est}_{stride}".format(est=est, stride=stride))

        #os.system("bash launch_slurm_job.sh p100 global_infomax_encoder_${est}_${stride} 1 "
        #          "python3 deep_infomax.py --encoder_stride ${stride} --mi_estimator ${est} --gpu "
        #          "--prefix global_infomax_encoder_${est}_${stride} --global_dim")

