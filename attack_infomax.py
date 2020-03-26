import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.classifier import *
from utils.argparser import argparser
from utils import data_loaders
from attacks.evaluation import evaluate_adversarial
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

    classifier = classifier.to(args.device)
    evaluate_adversarial(args, classifier, test_loader)

