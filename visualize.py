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
    :param args:
    :return:
    """

    # pass clean X through classifier's encoder, get clean representations
    C_clean, E_clean = classifier.encoder(X)
    print(E_clean.shape)

    # compute adversarial perturbations and save
    # X_adv, delta, out, out_adv = pgd(model=classifier, X=X, y=Y, epsilon=args.epsilon,
    #                                  alpha=args.alpha, num_steps=args.num_steps, random_restart=False, p="inf")

    X_adv, delta, out, out_adv = fgsm(classifier, X, Y, args.epsilon)
    X = ((X * std + mean) * 255).int()

    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()

    plt.imshow(delta[0].permute(1, 2, 0) * 50)
    plt.show()

    # pass perturbed input through classifier's encoder, get perturbed representations
    C_adv, E_adv = classifier.encoder(X_adv)

    X_adv = ((X_adv * std + mean) * 255).int()
    plt.imshow(X_adv[0].permute(1, 2, 0))
    plt.show()

    X_hat = decoder(E_clean.unsqueeze(0))
    X_hat = ((X_hat * std + mean) * 255).int()
    plt.imshow(X_hat[0].permute(1, 2, 0))
    plt.show()

    X_hat_adv = decoder(E_adv.unsqueeze(0))
    X_hat_adv = ((X_hat_adv * std + mean) * 255).int()
    plt.imshow(X_hat_adv[0].permute(1, 2, 0))
    plt.show()

    # compare clean vs pert representations (L2 norm, difference per dimension, are some dimensions consistently unchanged?)
    C_l2 = torch.frobenius_norm(C_adv - C_clean, dim=(-1, -2))
    E_l2 = torch.norm(E_clean - E_adv, p=2, dim=-1, keepdim=True)

    print("C_l2", C_l2)
    print("E_l2", E_l2)
    print(out, out_adv)

if __name__ == "__main__":

    args = argparser()

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
    classifier.load_state_dict(torch.load("checkpoints/cifar10_classification_checkpoint.pth",
                                          map_location=torch.device("cpu"))["classifier_state_dict"])

    decoder = DecoderY(input_size=encoder.output_size)
    decoder.load_state_dict(torch.load("checkpoints/cifar10_decoding_checkpoint.pth",
                                       map_location=torch.device("cpu"))["decoder_state_dict"])
    decoder.eval()

    classifier = classifier.to(args.device)

    for X, Y in test_loader:
        visualize_attacks(classifier, decoder, X, Y, args)







