import torch
import torch.nn as nn
from attacks import evaluation
from attacks.gradient_untargeted import pgd, fgsm
from utils.argparser import argparser
from utils import data_loaders
from matplotlib import pyplot as plt
from models.encoders import *
from models.classifier import *
from models.decoder import *
import random
import numpy as np

mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
std = torch.tensor([0.225, 0.225, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def visualize_attacks(classifier, decoder, X, Y, args):
    """
    Generate attacks for X according to args, save images, perturbations and predictions
    :param classifier
    :param X:
    :param Y:
    :param args: c
    :return:
    """

    # pass clean X through classifier's encoder, get clean representations
    Z_clean = classifier.encoder(X, intermediate=True)
    print(Z_clean.shape)

    # compute adversarial perturbations and save
    X_adv, delta, out, out_adv = pgd(model=classifier, X=X, y=Y, epsilon=args.epsilon,
                                     alpha=args.alpha, num_steps=args.num_steps, random_restart=False, p="inf")

    # X_adv, delta, out, out_adv = fgsm(classifier, X, Y, args.epsilon)
    X = ((X * std + mean) * 255).int()

    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()

    plt.imshow(delta[0].detach().permute(1, 2, 0) * 50)
    plt.show()

    # pass perturbed input through classifier's encoder, get perturbed representations
    Z_adv = classifier.encoder(X_adv, intermediate=True)

    X_adv = ((X_adv * std + mean) * 255).int()
    plt.imshow(X_adv[0].permute(1, 2, 0))
    plt.show()

    # X_hat = decoder(E_clean)
    # X_hat = ((X_hat * std + mean) * 255).int()
    # plt.imshow(X_hat[0].permute(1, 2, 0))
    # plt.show()
    #
    # X_hat_adv = decoder(E_adv)
    # X_hat_adv = ((X_hat_adv * std + mean) * 255).int()
    # plt.imshow(X_hat_adv[0].permute(1, 2, 0))
    # plt.show()

    # compare clean vs pert representations (L2 norm, difference per dimension, are some dimensions consistently unchanged?)
    Z_l2 = torch.norm(Z_clean - Z_adv, p=2, dim=-1, keepdim=True)
    print("Z_l2", Z_l2, Z_clean, Z_adv)

    # reshape E x E
    E_diff = torch.abs((Z_adv - Z_clean)).reshape(-1, 32, 32)
    plt.matshow(E_diff[0].detach().numpy())
    plt.show()

    print(out, out_adv)

if __name__ == "__main__":

    args = argparser()

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
    # classifier.load_state_dict(torch.load("classifier_supervised_new/classifier_supervised_new_checkpoint.pth",
    #                                       map_location=torch.device("cpu"))["classifier_state_dict"])

        classifier.load_state_dict(torch.load("classifier_fc_jsd_prior/classifier_fc_jsd_prior_checkpoint.pth",
                                          map_location=torch.device("cpu"))["classifier_state_dict"])

    # decoder = DecoderY(input_size=encoder.output_size)
    # decoder.load_state_dict(torch.load("decoder_jsd_new/decoder_jsd_new_checkpoint.pth",
    #                                    map_location=torch.device("cpu"))["decoder_state_dict"])
    # decoder.eval()

    classifier = classifier.to(args.device)

    #evaluation.evaluate_adversarial(args, model=classifier, loader=test_loader)
    for X, Y in test_loader:
        visualize_attacks(classifier, None, X, Y, args)







