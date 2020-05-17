import torch
import torch.nn as nn
import os
from attacks.gradient_untargeted import pgd, fgsm
from attacks.mi_attacks import *
from utils.argparser import argparser
from utils import data_loaders
from models.classifier import *
from models.decoder import *
import random
from tqdm import tqdm
import numpy as np
import time
from utils.train_eval import AverageMeter


def get_encoder_transfer_stats(args, source_model, target_model, loader, log):

    source_z_l2_norms = AverageMeter()
    target_z_l2_norms = AverageMeter()

    source_model.eval()
    target_model.eval()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        _, _, E = source_model(X)
        source_x_adv, source_e_adv, source_diff, source_max_diff = encoder_attack(X, source_model, args.num_steps, args.epsilon, args.alpha,
                                                                                  random_restart=True)

        source_z_l2_norms.update(source_diff)
        _, _, E_adv = target_model(source_x_adv)
        _, _, E = target_model(X)
        l2 = torch.norm(E - E_adv, dim=-1, p=2).mean()
        target_z_l2_norms.update(l2)

        print("Src L2 {src_l2:3f}\t"
              "Tgt L2 {tgt_l2:3f}\t", file=log)


def get_classifier_transfer_stats(args, source_model, target_model, loader, log):

    source_clean_errors = AverageMeter()
    source_adv_errors = AverageMeter()
    target_clean_errors = AverageMeter()
    target_adv_errors = AverageMeter()

    # compute the attack success rate: fraction of changed predictions
    source_success_changed_meter = AverageMeter()
    target_success_changed_meter = AverageMeter()

    # compute the attack success rate: fraction of changed predictions from correct predictions
    source_success_correct_meter = AverageMeter()
    target_success_correct_meter = AverageMeter()

    # compute the fraction of successfully transferred attacks
    fraction_transfer_meter = AverageMeter()

    # compute the fraction of successfully transferred attacks with the same prediction
    fraction_transfer_same_meter = AverageMeter()

    source_model.eval()
    target_model.eval()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        # adv samples
        if args.attack == "pgd":
            X_adv, delta, source_out, source_out_adv = pgd(model=source_model, X=X, y=y, epsilon=args.epsilon,
                                             alpha=args.alpha, num_steps=args.num_steps, p='inf')

        elif args.attack == "fgsm":
            X_adv, delta, source_out, source_out_adv = fgsm(model=source_model, X=X, y=y, epsilon=args.epsilon)

        err_clean = (source_out.data != y).float().sum() / X.size(0)
        err_adv = (source_out_adv.data != y).float().sum() / X.size(0)

        # compute success rate overall
        source_success_all = (source_out_adv.data != source_out.data).float()

        # compute success rate for correctly classified inputs
        source_mask = (source_out.data == y).float()
        source_success_correct = ((source_out_adv.data != source_out.data).float() * source_mask)

        # update source model stats
        source_clean_errors.update(err_clean)
        source_adv_errors.update(err_adv)
        source_success_changed_meter.update(source_success_all.sum() / X.size(0))
        source_success_correct_meter.update(source_success_correct.sum() / source_mask.sum())

        # pass X_adv to target model and compute success rate
        target_out = target_model(X)
        target_out_adv = target_model(X_adv)
        err_clean = (target_out.data != y).float().sum() / X.size(0)
        err_adv = (target_out_adv.data != y).float().sum() / X.size(0)

        # compute success rate overall
        target_success_all = (target_out_adv.data != target_out.data).float()

        # compute success rate for correctly classified inputs
        target_mask = (target_out.data == y).float()
        target_success_correct = ((target_out_adv.data != target_out.data).float() * target_mask)

        # update target model stats
        target_clean_errors.update(err_clean)
        target_adv_errors.update(err_adv)
        target_success_changed_meter.update(target_success_all.sum() / X.size(0))
        target_success_correct_meter.update(target_success_correct.sum() / target_mask.sum())

        # compute fraction of successfully transferred attacks
        fraction_transfer_meter.update((source_success_all * target_success_all).float().sum() /  source_success_all.sum())

        # compute fraction of matched misclassifications from successfully transferred attacks
        fraction_transfer_same_meter.update((source_success_all * target_success_all).float() * \
        (target_out_adv == source_out_adv) / (source_success_all * target_success_all).float().sum())

        batch.set_description("Adv Error{}, Transfer Rate {} Same Pred Rate {}".format(source_adv_errors,
                                                                                       fraction_transfer_meter.avg,
                                                                                       fraction_transfer_same_meter))


    print('Source Clean Error {source_clean_error.avg:.3f}\t'
          'Source Adv Error {source_adv_error.avg:.3f}\t'
          'Target Clean Error {target_clean_error.avg:.3f}\t'
          'Target Adv Error {target_adv_error.avg:.3f}\t'
          'Source Changed {source_success_changed.avg:.3f}\t'
          'Target Changed {target_success_changed.avg:.3f}\t'
          'Source Correct Changed {source_success_correct:.3f}\t'
          'Target Correct Changed {target_success_correct:.3f}\t'
          'Fraction Trasfer {fraction_transfer:.3f}\t'
          'Fraction Transfer Same {fraction_transfer_same:.3f}\t'
          .format(source_clean_error=source_clean_errors, source_adv_error=source_adv_errors,
                  target_clean_error=target_clean_errors, target_adv_error=target_adv_errors,
                  source_success_changed=source_success_changed_meter,
                  target_success_changed=target_success_changed_meter,
                  source_success_correct=source_success_correct_meter,
                  target_success_correct=target_success_correct_meter,
                  fraction_transfer=fraction_transfer_meter,
                  fraction_transfer_same=fraction_transfer_same_meter), file=log)

def load_classifier(model_ckpt, args):
    encoder = GlobalEncoder(stride=args.encoder_stride)

    # create classifiers
    if args.input_layer == "fc":
        classifier = ClassifierFC(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "conv":
        classifier = ClassifierConv(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "y":
        classifier = ClassifierY(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    classifier.load_state_dict(torch.load(model_ckpt,
                                          map_location=args.device)["classifier_state_dict"])

    classifier = classifier.to(args.device)
    return classifier


def load_encoder(model_ckpt, args):
    encoder = GlobalEncoder(stride=args.encoder_stride)
    encoder.load_state_dict(torch.load(model_ckpt,
                                       map_location=args.device)["encoder_state_dict"])
    return encoder

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace, note that for transfer evaluations, a new workspace is created
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    stats_log = open("{}/{}".format(workspace_dir, args.log), "w")
    # write source and target models in transfer stats log
    print("Source Model: {}\t Target Model: {}\t".format(args.source_model_ckpt, args.target_model.ckpt))

    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    # TODO add option for classifiers
    source_model = load_encoder(args.source_model_ckpt, args)
    target_model = load_encoder(args.target_model_ckpt, args)

    get_encoder_transfer_stats(args, source_model, target_model, test_loader, log=stats_log)

