import torch
import torch.nn as nn
from attacks import evaluation
from attacks.gradient_untargeted import pgd, fgsm
from attacks.mi_attacks import *
from utils.argparser import argparser
from utils import data_loaders
from matplotlib import pyplot as plt
from models.encoders import *
from models.classifier import *
from models.decoders import *
import random
import numpy as np


def visualize_encoder_attacks(encoder, decoder, X, Y, args):

    # pass clean X through classifier's encoder, get clean representations
    C_clean, FC_clean, Z_clean = encoder(X)
    print(Z_clean.shape)

    X_adv, E_adv, diff, max_diff = encoder_attack(X, encoder, args.num_steps, args.epsilon, args.alpha,
                                                  random_restart=True)

    delta = X_adv - X

    # X_adv, delta, out, out_adv = fgsm(classifier, X, Y, args.epsilon)
    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()

    plt.imshow(delta[0].detach().permute(1, 2, 0) * 50)
    plt.show()

    plt.imshow(X_adv[0].detach().permute(1, 2, 0))
    plt.show()

    X_hat = decoder(Z_clean)
    plt.imshow(X_hat[0].detach().permute(1, 2, 0))
    plt.show()

    print(E_adv.shape)
    X_hat_adv = decoder(E_adv)
    plt.imshow(X_hat_adv[0].detach().permute(1, 2, 0))
    plt.show()

    # compare clean vs pert representations (L2 norm, difference per dimension, are some dimensions consistently unchanged?)
    Z_l2 = torch.norm(Z_clean - E_adv, p=2, dim=-1, keepdim=True)
    print("Z_l2", Z_l2, Z_clean, E_adv)

    # reshape E x E
    E_diff = torch.abs((E_adv - Z_clean) / Z_clean).reshape(-1, 8, 8)
    plt.matshow(E_diff[0].detach().numpy())
    plt.show()

def visualize_impostor_attacks(encoder, decoder, X, y, args):

    batch_size = X.shape[0]
    print(batch_size)
    # using the given batch form X_s X_t pairs
    X_s = X[0:batch_size // 2]
    X_t = X[batch_size // 2:]
    # set y to the labels for X_s and X to X_s for later computation and logging
    y = y[0:batch_size // 2]
    C_clean, FC_clean, Z_clean = encoder(X)
    print(X.shape, Z_clean.shape)
    X_adv, E_adv, diff, min_diff = source2target(X_s, X_t, encoder=encoder, epsilon=0.03,
                                                 max_steps=770, step_size=0.0001)

    print("Avg Diff {} Min Diff {}".format(diff, min_diff))
    delta = X_adv - X

    # X_adv, delta, out, out_adv = fgsm(classifier, X, Y, args.epsilon)
    X = ((X * std + mean) * 255).int()

    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()

    plt.imshow(X[1].permute(1, 2, 0))
    plt.show()

    plt.imshow(delta[0].detach().permute(1, 2, 0) * 50)
    plt.show()

    X_adv = ((X_adv * std + mean) * 255).int()
    plt.imshow(X_adv[0].permute(1, 2, 0))
    plt.show()

    if decoder is not None:
        X_hat = decoder(Z_clean)
        X_hat = ((X_hat * std + mean) * 255).int()
        plt.imshow(X_hat[0].permute(1, 2, 0))
        plt.show()

        print(E_adv.shape)
        X_hat_adv = decoder(E_adv.unsqueeze(0))
        X_hat_adv = ((X_hat_adv * std + mean) * 255).int()
        plt.imshow(X_hat_adv[0].permute(1, 2, 0))
        plt.show()

    # reshape E x E
    E_diff = torch.abs((E_adv - Z_clean) / Z_clean).reshape(-1, 8, 8)
    plt.matshow(E_diff[0].detach().numpy())
    plt.show()

if __name__ == "__main__":

    args = argparser()

    train_loader, _ = data_loaders.celeb_loaders(args.batch_size)
    _, test_loader = data_loaders.celeb_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride, output_size=args.code_size, input_size=64)
    encoder.load_state_dict(torch.load("checkpoints/celeb/local_infomax_encoder_dv_celeb/"
                                       "local_infomax_encoder_dv_celeb_checkpoint.pth",
                                       map_location=torch.device("cpu"))["encoder_state_dict"])
    encoder.eval()

    decoder = DecoderY(input_size=encoder.output_size, output_size=64)
    decoder.load_state_dict(torch.load("experiments/"
                                       "decoder_local_infomax_encoder_dv_celeb/decoder_local_infomax_encoder_dv_celeb_checkpoint.pth",
                                       map_location=torch.device("cpu"))["decoder_state_dict"])
    decoder.eval()

    #evaluation.evaluate_adversarial(args, model=classifier, loader=test_loader)
    for X, Y in test_loader:
        visualize_encoder_attacks(encoder, decoder, X, Y, args)

