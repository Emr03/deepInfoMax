import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.classifier import *
from utils.argparser import argparser
from utils import data_loaders
from attacks.evaluation import evaluate_adversarial
from attacks.gradient_untargeted import pgd, fgsm
from utils.train_eval import AverageMeter
import random
import numpy as np
import json
from tqdm import tqdm


def get_attack_stats(args, classifier, discriminator, loader, log):

    clean_errors = AverageMeter()
    adv_errors = AverageMeter()

    classifier.eval()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        Z_clean = classifier.encoder(X, intermediate=True)

        # adv samples using classifier
        if args.attack == "pgd":
            X_adv, delta, out, out_adv = pgd(model=classifier, X=X, y=y, epsilon=args.epsilon,
                                             alpha=args.alpha, num_steps=args.num_steps, p='inf')

        elif args.attack == "fgsm":
            X_adv, delta, out, out_adv = fgsm(model=classifier, X=X, y=y, epsilon=args.epsilon)

        err_clean = (out.data != y).float().sum() / X.size(0)
        err_adv = (out_adv.data != y).float().sum() / X.size(0)

        clean_errors.update(err_clean)
        adv_errors.update(err_adv)

        # evaluate the critic scores for X and E
        mi, E, scores = discriminator(X=X, return_scores=True)

        # evaluate the critic scores for X_adv and E_adv
        mi_adv_adv, E_adv, scores_adv_adv = discriminator(X=X_adv, return_scores=True)

        # evaluate the critic scores for X_adv and E_clean
        mi_adv_clean, _, scores_adv_clean = discriminator(X_adv, E=E, return_scores=True)

        # evaluate the critic scores for X, E_adv
        mi_clean_adv, _, scores_clean_adv = discriminator(X, E=E_adv, return_scores=True)

        batch.set_description("MI(X, E) {} MI(X_adv, E_adv) {} MI(X_adv, E) {} MI(X, E_adv) {}".format(mi, mi_adv_adv,
                                                                                                       mi_adv_clean,
                                                                                                       mi_clean_adv))
        # print to logfile
        print("Error Clean {} Error Adv{}, MI(X, E) {} MI(X_adv, E_adv) {} MI(X_adv, E) {} MI(X, E_adv) {}".format(
            clean_errors.avg, adv_errors.avg, mi, mi_adv_adv, mi_adv_clean, mi_clean_adv), file=log)


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    attack_log = open("{}/attack.log".format(workspace_dir), "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)

    # create classifier
    if args.input_layer == "fc":
        classifier = ClassifierFC(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "conv":
        classifier = ClassifierConv(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "y":
        classifier = ClassifierY(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    # load classifier from checkpoint
    classifier.load_state_dict(torch.load(args.classifier_ckpt)["classifier_state_dict"])

    # if args.cuda_ids and len(args.cuda_ids) > 1:
    #     classifier = nn.DataParallel(classifier)

    DIM = LocalDIM(encoder, type=args.mi_estimator)
    DIM = nn.DataParallel(DIM).to(args.device)
    DIM.module.T.load_state_dict(torch.load(args.encoder_ckpt)["discriminator_state_dict"])
    classifier = classifier.to(args.device)
    discriminator = DIM.module
    get_attack_stats(args, classifier, discriminator, test_loader, log=attack_log)

