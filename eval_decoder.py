import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.decoder import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
import random
import numpy as np
import json

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    # save arguments as json file
    # json.dump(obj=args, separators="\t", indent=4, fp="{}_args".format(workspace_dir))

    #train_log = open("{}/train.log".format(workspace_dir), "w")
    test_log = open("{}/test.log".format(workspace_dir), "w")

    #train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    # load encoder from checkpoint
    encoder.load_state_dict(torch.load(args.encoder_ckpt)["encoder_state_dict"])
    encoder = encoder.to(args.device)

    decoder = DecoderY(input_size=encoder.output_size)
    decoder = decoder.to(args.device)

    ms_ssim = train_eval.eval_decoder(test_loader, encoder, decoder, log=test_log, verbose=True, gpu=args.gpu)
