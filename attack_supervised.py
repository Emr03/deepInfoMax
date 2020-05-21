import torch
import torch.nn as nn
from utils.data_loaders import *
from models.classifier import *
from utils.argparser import argparser
from utils import data_loaders
from attacks.evaluation import evaluate_adversarial
from attacks.gradient_untargeted import pgd, fgsm
from attacks.mi_attacks import encoder_attack, source2target
from utils.train_eval import AverageMeter
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import random
import numpy as np
import json
from tqdm import tqdm
import seaborn as sns


def get_attack_stats(args, encoder, classifier, loader, log, type="class"):
    clean_errors = AverageMeter()
    adv_errors = AverageMeter()

    c_l2_norms = AverageMeter()
    c_l2_frac = AverageMeter()

    fc_l2_norms = AverageMeter()
    fc_l2_frac = AverageMeter()

    z_l2_norms = AverageMeter()
    z_l2_frac = AverageMeter()

    classifier.eval()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        if type == "class":
            # adv samples using classifier
            if args.attack == "pgd":
                X_adv, delta, out, out_adv = pgd(model=classifier, X=X, y=y, epsilon=args.epsilon,
                                                 alpha=args.alpha, num_steps=args.num_steps, p='inf')

            elif args.attack == "fgsm":
                X_adv, delta, out, out_adv = fgsm(model=classifier, X=X, y=y, epsilon=args.epsilon)

        elif type == "encoder":
            X_adv, E_adv, diff, max_diff = encoder_attack(X, encoder, args.num_steps, args.epsilon, args.alpha,
                                                          random_restart=True)

            batch.set_description("Avg Diff {} Max Diff {}".format(diff, max_diff))

            # run classifier on adversarial representations
            logits_clean = classifier(X)
            logits_adv = classifier(X_adv)
            out = logits_clean.max(1)[1]
            out_adv = logits_adv.max(1)[1]

        elif type == "impostor":
            batch_size = X.shape[0]
            # using the given batch form X_s X_t pairs
            X_s = X[0:batch_size // 2]
            X_t = X[batch_size // 2:]
            # set y to the labels for X_s and X to X_s for later computation and logging
            y = y[0:batch_size // 2]
            X = X_s

            X_adv, E_adv, diff, min_diff = source2target(X_s, X_t, encoder=encoder, epsilon=2.0,
                                                         max_steps=70000, step_size=0.001)

            # run classifier on adversarial representations
            logits_clean = classifier(X_s)
            logits_adv = classifier(X_adv)
            out = logits_clean.max(1)[1]
            out_adv = logits_adv.max(1)[1]

            batch.set_description("Avg Diff {} Min Diff {}".format(diff, min_diff))

        elif type == "random":

            delta = torch.rand_like(X).sign() * args.epsilon
            X_adv = X + delta
            _, _, E = encoder(X)
            _, _, E_d = encoder(X_adv)
            norm = torch.norm(E - E_d, p=2, dim=-1)

            # run classifier on adversarial representations
            logits_clean = classifier(X)
            logits_adv = classifier(X_adv)
            out = logits_clean.max(1)[1]
            out_adv = logits_adv.max(1)[1]

            batch.set_description("Avg Diff {} Max Diff {} ".format(norm.mean(), norm.max()))

        # UPDATE CLEAN and ADV ERRORS
        err_clean = (out.data != y).float().sum() / X.size(0)
        err_adv = (out_adv.data != y).float().sum() / X.size(0)
        clean_errors.update(err_clean)
        adv_errors.update(err_adv)

        # UPDATE L2 NORM METERS
        C_clean, FC_clean, Z_clean = encoder(X)
        C_adv, FC_adv, Z_adv = encoder(X_adv)

        l2 = torch.norm(Z_clean - Z_adv, p=2, dim=-1, keepdim=True)
        fraction = (l2 / torch.norm(Z_clean, p=2, dim=-1, keepdim=True))
        z_l2_norms.update(l2.mean())
        z_l2_frac.update(fraction.mean())

        l2 = torch.norm(C_clean - C_adv, p=2, dim=(-1, -2, -3), keepdim=True)
        fraction = (l2 / torch.norm(C_clean, p=2, dim=(-1, -2, -3), keepdim=True))
        c_l2_norms.update(l2.mean())
        c_l2_frac.update(fraction.mean())

        l2 = torch.norm(FC_clean - FC_adv, p=2, dim=-1, keepdim=True)
        fraction = (l2 / torch.norm(FC_clean, p=2, dim=-1, keepdim=True))
        fc_l2_norms.update(l2.mean())
        fc_l2_frac.update(fraction.mean())

        # print to logfile
        print("Error Clean {clean_errors.avg:.3f}\t Error Adv{adv_errors.avg:.3f}\t "
              "C L2 {c_l2_norms.avg:.3f}\t C L2 Frac{c_l2_frac.avg:.3f}\t"
              "FC L2 {fc_l2_norms.avg:.3f}\t FC L2 Frac{fc_l2_frac.avg:.3f}\t"
              "Z L2 {z_l2_norms.avg:.3f}\t Z L2 Frac{z_l2_frac.avg:.3f}\t".format(
            clean_errors=clean_errors, adv_errors=adv_errors,
            c_l2_norms=c_l2_norms, c_l2_frac=c_l2_frac,
            fc_l2_norms=fc_l2_norms, fc_l2_frac=fc_l2_frac,
            z_l2_norms=z_l2_norms, z_l2_frac=z_l2_frac),
            file=log)

        log.flush()


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    class_attack_log = open("{}/class_attack.log".format(workspace_dir), "w")
    encoder_attack_log = open("{}/encoder_attack.log".format(workspace_dir), "w")
    impostor_attack_log = open("{}/impostor_attack.log".format(workspace_dir), "w")
    random_attack_log = open("{}/random_attack.log".format(workspace_dir), "w")

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
    classifier = classifier.to(args.device)
    get_attack_stats(args, classifier.encoder, classifier, test_loader, log=class_attack_log, type="class")
    get_attack_stats(args, classifier.encoder, classifier, test_loader, log=encoder_attack_log, type="encoder")
    get_attack_stats(args, classifier.encoder, classifier, test_loader, log=impostor_attack_log, type="impostor")
    get_attack_stats(args, classifier.encoder, classifier, test_loader, log=random_attack_log, type="random")


