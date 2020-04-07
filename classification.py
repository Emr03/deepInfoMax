import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.classifier import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
import random
import numpy as np
import json

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    # save arguments as json file
    # json.dump(obj=args, separators="\t", indent=4, fp="{}_args".format(workspace_dir))

    train_log = open("{}/train.log".format(workspace_dir), "w")
    test_log = open("{}/test.log".format(workspace_dir), "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    # load encoder from checkpoint
    encoder.load_state_dict(torch.load(args.encoder_ckpt)["encoder_state_dict"])
    # create classifier
    if args.input_layer == "fc":
        classifier = ClassifierFC(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "conv":
        classifier = ClassifierConv(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    elif args.input_layer == "y":
        classifier = ClassifierY(encoder=encoder, hidden_units=args.hidden_units, num_classes=10)

    # if args.cuda_ids and len(args.cuda_ids) > 1:
    #     classifier = nn.DataParallel(classifier)

    classifier = classifier.to(args.device)

    opt = optim.Adam(classifier.parameters(), lr=args.lr)

    # if num of visible devices > 1, use DataParallel wrapper
    e = 0
    while e < args.epochs:
        loss = train_eval.train_classifier(train_loader, classifier, opt, e,
                                           train_log, verbose=args.verbose, gpu=args.gpu)
        e += 1

        train_eval.eval_classifier(test_loader, classifier, e, test_log, verbose=args.verbose, gpu=args.gpu)
        torch.save({
            'classifier_state_dict': classifier.state_dict(),
            'epoch': e,
            'opt': opt.state_dict(),
            'loss': loss,
        }, args.prefix + "_checkpoint.pth")






