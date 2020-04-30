import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.classifier import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
import random
import numpy as np
import json
from attacks.evaluation import evaluate_adversarial

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    # save arguments as json file
    # json.dump(obj=args, separators="\t", indent=4, fp="{}_args".format(workspace_dir))

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    if not args.classifier_ckpt:
        encoder = GlobalEncoder(stride=args.encoder_stride)

        # open logs in write mode
        train_log = open("{}/train.log".format(workspace_dir), "w")
        test_log = open("{}/test.log".format(workspace_dir), "w")

        if not args.fully_supervised:
            # load encoder from checkpoint
            encoder.load_state_dict(torch.load(args.encoder_ckpt)["encoder_state_dict"])

    # create classifier
    freeze_encoder = not args.fully_supervised or args.random_encoder
    if args.input_layer == "fc":
        classifier = ClassifierFC(encoder=encoder, dropout=args.dropout, hidden_units=args.hidden_units, num_classes=10,
                                  freeze_encoder=freeze_encoder)

    elif args.input_layer == "conv":
        classifier = ClassifierConv(encoder=encoder, dropout=args.dropout, hidden_units=args.hidden_units, num_classes=10,
                                    freeze_encoder=freeze_encoder)

    elif args.input_layer == "y":
        classifier = ClassifierY(encoder=encoder, dropout=args.dropout, hidden_units=args.hidden_units, num_classes=10,
                                 freeze_encoder=freeze_encoder)

    classifier = classifier.to(args.device)
    opt = optim.Adam(classifier.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    e = 0
    test_err = 1.0

    if args.classifier_ckpt:
        # open logs in append mode
        train_log = open("{}/train.log".format(workspace_dir), "a")
        test_log = open("{}/test.log".format(workspace_dir), "a")

        # resume training
        ckpt = torch.load(args.classifier_ckpt)
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        opt.load_state_dict(ckpt["opt"])
        e = ckpt["epoch"]
        test_err = ckpt["test_err"]

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[15, 50, 100, 200, 300], gamma=0.5)
    # if num of visible devices > 1, use DataParallel wrapper
    while e < args.epochs:
        if args.classifier_adversarial:
            loss = train_eval.train_classifier_adversarial(train_loader, classifier, opt, e, train_log,
                                                           verbose=args.verbose, gpu=args.gpu, args=args)

            clean_errors, adv_errors = evaluate_adversarial(args, classifier, test_loader)
            test_err_ = adv_errors
            

        else:
            loss = train_eval.train_classifier(train_loader, classifier, opt, e,
                                               train_log, verbose=args.verbose, gpu=args.gpu)

            test_err_ = train_eval.eval_classifier(test_loader, classifier, e, test_log, verbose=args.verbose, gpu=args.gpu)

        #scheduler.step()
        e += 1
        print(test_err_)
        if test_err > test_err_:
            test_err = test_err_
            torch.save({
                'classifier_state_dict': classifier.state_dict(),
                'epoch': e,
                'opt': opt.state_dict(),
                'test_err': test_err,
            }, workspace_dir + "/" + args.prefix + "_checkpoint.pth")






