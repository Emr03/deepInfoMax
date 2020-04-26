import torch
import torch.nn as nn
import os
from attacks.gradient_untargeted import pgd, fgsm
from utils.argparser import argparser
from utils import data_loaders
from models.classifier import *
from models.decoder import *
import random
from tqdm import tqdm
import numpy as np
import time
from utils.train_eval import AverageMeter

def get_attack_stats(args, model, loader, log):

    batch_time = AverageMeter()
    clean_errors = AverageMeter()
    adv_errors = AverageMeter()
    l2_norms = AverageMeter() # l2 norm between representations for clean input and adversarial input, proxy for stability
    change_fraction = AverageMeter()

    model.eval()
    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        Z_clean = classifier.encoder(X, intermediate=True)

        # adv samples
        if args.attack == "pgd":
            X_adv, delta, out, out_adv = pgd(model=model, X=X, y=y, epsilon=args.epsilon,
                                             alpha=args.alpha, num_steps=args.num_steps, p='inf')

        elif args.attack == "fgsm":
            X_adv, delta, out, out_adv = fgsm(model=model, X=X, y=y, epsilon=args.epsilon)
        
        err_clean = (out.data != y).float().sum() / X.size(0)
        err_adv = (out_adv.data != y).float().sum() / X.size(0)

        clean_errors.update(err_clean)
        adv_errors.update(err_adv)

        # pass perturbed input through classifier's encoder, get perturbed representations
        Z_adv = classifier.encoder(X_adv, intermediate=True)

        Z_l2 = torch.norm(Z_clean - Z_adv, p=2, dim=-1, keepdim=True).mean()
        l2_norms.update(Z_l2)

        # compute fraction of l1_norm
        fraction = (torch.abs(Z_clean - Z_adv) / Z_clean)
        print(fraction.shape)
        print(fraction.max())
        change_fraction.update(fraction.max())

        batch.set_description("Clean Err {} Adv Err {} L2 {} Frac {}".format(clean_errors.avg, adv_errors.avg,
                                                                             l2_norms.avg, change_fraction.avg))

        # print to logfile
        print("clean_err: ", clean_errors.avg,
              " adv_err: ", adv_errors.avg,
              "l2 norm: ", l2_norms.avg,
              "l1 frac: ", change_fraction.avg,
              file=log)

    print(' * Clean Error {clean_error.avg:.3f}\t'
          ' Adv Error {adv_errors.avg:.3f}\t'
          ' L2 norm {l2_norms.avg:.3f}\t'
          ' L1 frac {change_frac.avg:.3f}\t'
          .format(clean_error=clean_errors, adv_errors=adv_errors,
                  l2_norms=l2_norms, change_frac=change_fraction))


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    stats_log = open("{}/stats.log".format(workspace_dir), "w")

    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)

    # create classifiers
    if args.input_layer == "fc":
        classifier = ClassifierFC(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "conv":
        classifier = ClassifierConv(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "y":
        classifier = ClassifierY(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    classifier.load_state_dict(torch.load(args.classifier_ckpt,
                                      map_location=args.device)["classifier_state_dict"])

    classifier = classifier.to(args.device)

    get_attack_stats(args, classifier, test_loader, log=stats_log)
