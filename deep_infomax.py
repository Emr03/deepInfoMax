import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.prior_matching import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
from utils.get_config import get_config
import random
import numpy as np
import os
import json

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    train_log = open("{}/train.log".format(workspace_dir), "w")
    test_log = open("{}/test.log".format(workspace_dir), "w")

    input_size, ndf, num_channels, train_loader, test_loader = get_config(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(ndf=ndf, num_channels=num_channels,
            output_size=args.code_size, input_size=input_size)

    if args.global_dim:
        DIM = GlobalDIM(encoder, type=args.mi_estimator)

    else:
        DIM = LocalDIM(encoder, type=args.mi_estimator)

    if args.prior_matching:
        prior_matching = PriorMatchingDiscriminator(encoder_dim=args.code_size, device=args.device)
        D_opt = optim.Adam(prior_matching.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        prior_matching = prior_matching.to(args.device) 

    else:
        prior_matching = None
        D_opt = None

    DIM = nn.DataParallel(DIM)
    DIM = DIM.to(args.device)

    enc_opt = optim.Adam(DIM.module.global_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T_opt = optim.Adam(DIM.module.T.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if num of visible devices > 1, use DataParallel wrapper
    e = 0
    while e < args.epochs:
        loss = train_eval.train_dim(train_loader, DIM, enc_opt, T_opt, e, train_log, args.verbose, args.gpu,
                                    prior_matching, D_opt, gamma=args.gamma)

        e += 1
        torch.save({
            'encoder_state_dict': DIM.module.global_encoder.state_dict(),
            'discriminator_state_dict': DIM.module.T.state_dict(),
            'prior_matching_state_dict': prior_matching.state_dict() if args.prior_matching else None,
            'epoch': e,
            'enc_opt': enc_opt.state_dict(),
            'T_opt': T_opt.state_dict(),
            'D_opt': D_opt.state_dict() if args.prior_matching else None,
            'loss': loss,
        }, workspace_dir + "/" + args.prefix + "_checkpoint.pth")

