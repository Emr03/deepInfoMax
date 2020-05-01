import torch
import torch.nn as nn
import os
from attacks.gradient_untargeted import pgd, fgsm
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
from models.encoders import *
from models.ndm import *
import torch.optim as optim
import random

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    ndm_train_log = open("{}/ndm_train.log".format(workspace_dir), "w")
    ndm_eval_log = open("{}/ndm_test.log".format(workspace_dir), "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    encoder.load_state_dict(torch.load(args.encoder_ckpt,
                                          map_location=args.device)["encoder_state_dict"])

    ndm_disc = NeuralDependencyMeasure(encoder=encoder)
    ndm_disc = ndm_disc.to(args.device)

    opt = optim.Adam(ndm_disc.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    e = 0
    while e < args.epochs:
        loss = train_eval.ndm_train(model=ndm_disc, loader=train_loader, opt=opt, epoch=e, verbose=args.verbose,
                                    log=ndm_train_log, gpu=args.gpu)

        loss = train_eval.ndm_eval(model=ndm_disc, loader=test_loader, log=ndm_eval_log, gpu=args.gpu)
        e += 1
        torch.save({
            'ndm_state_dict': ndm_disc.state_dict(),
            'epoch': e,
            'ndm_opt': opt.state_dict(),
            'loss': loss,
        }, workspace_dir + "/" + "ndm_checkpoint.pth")



