import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.decoders import *
from utils.get_config import get_config
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

    test_log = open("{}/test.log".format(workspace_dir), "w")

    input_size, _, test_loader = get_config(args.data)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride, input_size=input_size, output_size=args.code_size)
    # load encoder from checkpoint
    encoder.load_state_dict(torch.load(args.encoder_ckpt)["encoder_state_dict"])
    encoder = encoder.to(args.device)

    decoder = DeconvDecoder(input_size=encoder.output_size, output_size=input_size)
    decoder = decoder.to(args.device)

    ms_ssim = train_eval.eval_decoder(test_loader, encoder, decoder, log=test_log, verbose=True, gpu=args.gpu)
