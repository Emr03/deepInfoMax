import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
import random
import numpy as np

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder()
    if args.global_dim:
        DIM = GlobalDIM(encoder)

    else:
        DIM = LocalDIM(encoder)

    if len(args.cuda_ids) > 1:
        DIM = nn.DataParallel(DIM)

    DIM = DIM.to(args.device)
    enc_opt = optim.Adam(DIM.global_encoder.parameters(), lr=args.lr)
    T_opt = optim.Adam(DIM.T.parameters(), lr=args.lr)
    # if num of visible devices > 1, use DataParallel wrapper
    e = 0
    while e < args.epochs:
        loss = train_eval.train_dim(train_loader, DIM, enc_opt, T_opt, e, train_log, args.verbose, args.gpu)
        e += 1
        torch.save({
            'encoder_state_dict': DIM.global_encoder.state_dict(),
            'discriminator_state_dict': DIM.T.state_dict(),
            'epoch': e,
            'enc_opt': enc_opt.state_dict(),
            'T_opt': T_opt.state_dict(),
            'loss': loss,
        }, args.prefix + "_checkpoint.pth")






