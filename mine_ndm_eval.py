import torch
import torch.nn as nn
import os
from attacks.gradient_untargeted import pgd, fgsm
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
from models.encoders import *
from models.ndm import *
from models.mi_estimation import *
import torch.optim as optim
import random

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    ndm_train_log = open("{}/ndm_train.log".format(workspace_dir), "a")
    ndm_eval_log = open("{}/ndm_test.log".format(workspace_dir), "a")

    mine_train_log = open("{}/mine_train.log".format(workspace_dir), "a")
    mine_eval_log = open("{}/mine_test.log".format(workspace_dir), "a")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    encoder.load_state_dict(torch.load(args.encoder_ckpt,
                                          map_location=args.device)["encoder_state_dict"])

    # create Infomax module with loaded encoder, discriminator will be used to estimate mine
    DIM = LocalDIM(encoder, type="mine")
    DIM = DIM.to(args.device)
    ndm_disc = NeuralDependencyMeasure(encoder=encoder)
    ndm_disc = ndm_disc.to(args.device)

    ndm_opt = optim.Adam(ndm_disc.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mine_opt = optim.Adam(DIM.T.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    e = 0
   
    if args.mine_ckpt:
        ckpt = torch.load(args.mine_ckpt)
        DIM.T.load_state_dict(ckpt["mine_state_dict"])
        mine_opt.load_state_dict(ckpt["mine_opt"])
        e = ckpt["epoch"]
        ndm_disc.load_state_dict(ckpt["ndm_state_dict"])
        ndm_opt.load_state_dict(ckpt["ndm_opt"])

    while e < args.epochs:
        train_eval.mine_train(model=DIM, loader=train_loader, opt=mine_opt, log=mine_train_log, epoch=e,
                                   gpu=args.gpu, verbose=args.verbose)
        train_eval.ndm_train(model=ndm_disc, loader=train_loader, opt=ndm_opt, epoch=e, verbose=args.verbose,
                                    log=ndm_train_log, gpu=args.gpu)

        mi = train_eval.mine_eval(model=DIM, loader=test_loader, epoch=e, log=mine_eval_log, gpu=args.gpu)
        ndm = train_eval.ndm_eval(model=ndm_disc, loader=test_loader, epoch=e, log=ndm_eval_log, gpu=args.gpu)

        e += 1
        torch.save({
            'ndm_state_dict': ndm_disc.state_dict(),
            'mine_state_dict': DIM.T.state_dict(),
            'epoch': e,
            'ndm_opt': ndm_opt.state_dict(),
            'mine_opt': mine_opt.state_dict(),
            'ndm': ndm,
            'mi': mi
        }, workspace_dir + "/" + "mine_ndm_checkpoint.pth")





