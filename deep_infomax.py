import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.prior_matching import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
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

    # save arguments as json file
    #with open("{}_args".format(workspace_dir), "w") as f:
    #    json.dump(obj=args.__dict__, indent=4, fp=f)

    train_log = open("{}/train.log".format(workspace_dir), "w")
    test_log = open("{}/test.log".format(workspace_dir), "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    if args.global_dim:
        DIM = GlobalDIM(encoder, type=args.mi_estimator)

    else:
        DIM = LocalDIM(encoder, type=args.mi_estimator)

    if args.prior_matching:
        prior_matching = PriorMatchingDiscriminator(encoder_dim=64, device=args.device)
        D_opt = optim.Adam(prior_matching.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        prior_matching = prior_matching.to(args.device) 
        #prior_matching = nn.DataParallel(prior_matching)
    
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
        loss = train_eval.train_dim(train_loader, DIM, enc_opt, T_opt, e, train_log, args.verbose, args.gpu, prior_matching, D_opt, gamma=args.gamma)
        mi = train_eval.eval_dim(test_loader, DIM, e, test_log, args.verbose, args.gpu)

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
            'mi': mi,
        }, workspace_dir + "/" + args.prefix + "_checkpoint.pth")

