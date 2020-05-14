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

sns.set(style="ticks")

def get_attack_stats(args, classifier, discriminator, loader, log):

    clean_errors = AverageMeter()
    adv_errors = AverageMeter()

    mi_meter = AverageMeter()
    mi_adv_adv_meter = AverageMeter()
    mi_adv_clean_meter = AverageMeter()
    mi_clean_adv_meter = AverageMeter()

    classifier.eval()
    discriminator.eval()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

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

        with torch.no_grad():
            # evaluate the critic scores for X and E
            mi, E = discriminator(X=X)

            # evaluate the critic scores for X_adv and E_adv
            mi_adv_adv, E_adv = discriminator(X=X_adv)

            # evaluate the critic scores for X_adv and E_clean
            mi_adv_clean, _ = discriminator(X_adv, E=E)

            # evaluate the critic scores for X, E_adv
            mi_clean_adv, _ = discriminator(X, E=E_adv)


        batch.set_description("MI(X, E) {} MI(X_adv, E_adv) {} MI(X_adv, E) {} MI(X, E_adv) {}".format(mi, mi_adv_adv,
                                                                                                       mi_adv_clean,
                                                                                                       mi_clean_adv))

        # print to logfile
        print("Error Clean {} Error Adv{}, MI(X, E) {} MI(X_adv, E_adv) {} MI(X_adv, E) {} MI(X, E_adv) {}".format(
            clean_errors.avg, adv_errors.avg, mi, mi_adv_adv, mi_adv_clean, mi_clean_adv), file=log)

        mi_meter.update(mi)
        mi_adv_adv_meter.update(mi_adv_adv)
        mi_adv_clean_meter.update(mi_adv_clean)
        mi_clean_adv_meter.update(mi_clean_adv)


def attack_encoder(args, encoder, classifier, discriminator, loader, log):

    clean_errors = AverageMeter()
    adv_errors = AverageMeter()

    mi_meter = AverageMeter()
    mi_adv_adv_meter = AverageMeter()
    mi_adv_clean_meter = AverageMeter()
    mi_clean_adv_meter = AverageMeter()

    classifier.eval()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        X_adv, E_adv, diff, max_diff = encoder_attack(X, encoder, args.num_steps, args.epsilon, args.alpha,
                                                    random_restart=True)
        
        with torch.no_grad():
            # evaluate MI for X and X_adv
            mi, E = discriminator(X=X)

            # evaluate the critic scores for X_adv and E_adv
            mi_adv_adv, E_adv = discriminator(X=X_adv)

            # evaluate the critic scores for X_adv and E_clean
            mi_adv_clean, _ = discriminator(X_adv, E=E)

            # evaluate the critic scores for X, E_adv
            mi_clean_adv, _ = discriminator(X, E=E_adv)

        # run classifier on adversarial representations
        logits_clean = classifier(X)
        logits_adv = classifier(X_adv)
        out = logits_clean.max(1)[1]
        out_adv = logits_adv.max(1)[1]
        err_clean = (out.data != y).float().sum() / X.size(0)
        err_adv = (out_adv.data != y).float().sum() / X.size(0)

        # batch.set_description("MI(X, E) {} MI(X_adv, E_adv) {} MI(X_adv, E) {} MI(X, E_adv) {}".format(mi, mi_adv_adv,
        #                                                                                                mi_adv_clean,
        #                                                                                                mi_clean_adv))

        batch.set_description("Avg Diff {} Max Diff {}".format(diff, max_diff))
        mi_meter.update(mi)
        mi_adv_adv_meter.update(mi_adv_adv)
        mi_adv_clean_meter.update(mi_adv_clean)
        mi_clean_adv_meter.update(mi_clean_adv)

        clean_errors.update(err_clean)
        adv_errors.update(err_adv)

        # TODO decode
        # print to logfile
        print("Error Clean {}\t"
              "Error Adv{} \t"
              "MI(X, E) {} \t "
              "MI(X_adv, E_adv) {}\t"
              "MI(X_adv, E) {}\t"
              "MI(X, E_adv) {}\t"
              "Avg Diff {}\t"
              "Max Diff {}\t".format(
            clean_errors.avg, adv_errors.avg, mi, mi_adv_adv, mi_adv_clean, mi_clean_adv, diff, max_diff), file=log)


def impostor_attack(args, encoder, classifier, discriminator, loader, log):

    adv_diff_meter = AverageMeter()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        
        if args.gpu:
            X, y = X.cuda(), y.cuda()

        # using the given batch form X_s X_t pairs
        X_s = X[0:loader.batch_size // 2]
        X_t = X[loader.batch_size // 2:]

        X_adv, E_adv, diff, min_diff = source2target(X_s, X_t, encoder=encoder, epsilon=args.epsilon, step_size=0.001)
        adv_diff_meter.update(diff)
        batch.set_description("Avg Diff {} Min Diff {}".format(diff, min_diff))

        with torch.no_grad():
            # evaluate MI for X and X_adv
            mi, E = discriminator(X=X_s)

            # evaluate the critic scores for X_adv and E_adv
            mi_adv_adv, E_adv = discriminator(X=X_adv)

            # evaluate the critic scores for X_adv and E_clean
            mi_adv_clean, _ = discriminator(X_adv, E=E)

            # evaluate the critic scores for X, E_adv
            mi_clean_adv, _ = discriminator(X_s, E=E_adv)

        print("MI(X, E) {} \t "
              "MI(X_adv, E_adv) {}\t"
              "MI(X_adv, E) {}\t"
              "MI(X, E_adv) {}\t"
              "Avg Diff {}\t"
              "Min Diff {}\t".format(
            mi, mi_adv_adv, mi_adv_clean, mi_clean_adv, diff, min_diff), file=log)
              

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    class_attack_log = open("{}/class_attack.log".format(workspace_dir), "w")
    encoder_attack_log = open("{}/class_attack.log".format(workspace_dir), "w")
    impostor_attack_log = open("{}/impostor_attack.log".format(workspace_dir), "w")

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
    #get_attack_stats(args, classifier, discriminator, test_loader, log=class_attack_log)
    #attack_encoder(args, encoder, classifier, discriminator, test_loader, log=encoder_attack_log)
    impostor_attack(args, encoder, classifier, discriminator, test_loader, log=impostor_attack_log)
