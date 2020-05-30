import torch
import torch.nn as nn
from utils.matrix_utils import *
from torch.autograd import Variable
from utils.argparser import argparser
from utils.data_loaders import celeb_loaders
import os
import random
from models.encoders import *
from models.decoders import *
from models.vae import *
from matplotlib import pyplot as plt
from utils.train_eval import AverageMeter
from tqdm import tqdm

def get_projected_step(delta, g, p, epsilon, alpha):

    if p == 2:
        #print(g.shape, delta.shape)
        v = alpha * g / torch.norm(g, p=2, dim=(-1, -2, -3), keepdim=True).repeat(1, g.shape[-3], g.shape[-2], g.shape[-1])
        delta_norm = torch.norm(delta + v, p=2, dim=(-1, -2, -3), keepdim=True).repeat(1, g.shape[-3], g.shape[-2], g.shape[-1])
        # set delta norm to 1 / eps if norm > eps otherwise set to 1
        mask = (delta_norm.data > epsilon).float()
        delta_norm = (delta_norm * mask) / epsilon + delta_norm * (1 - mask) / delta_norm
        delta = (delta + v) / delta_norm

    elif p == 'inf':
        v = torch.sign(g) * alpha
        delta = torch.clamp(delta + v, min=-epsilon, max=epsilon)

    return Variable(delta.data, requires_grad=True)


def kl_div(mu, cov):
    return 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mu, p=2, dim=1))


def l2_wasserstein(mu_a, mu_b, cov_a, cov_b):
    """

    :param mu_a: batch of mean vectors
    :param mu_b: batch of mean vectors
    :param cov_a: batch of diagonal covariance metrices
    :param cov_b: batch of diagonal covariance matrices
    :return: l2 wasserstein distance between the Gaussian distributions defined by mu and cov at each batch element
    """
    first_term = torch.norm(mu_a - mu_b, p=2, dim=-1)
    cov_proj = torch.bmm(torch.pow(cov_b, 0.5), cov_a).bmm(torch.pow(cov_b, 0.5))
    second_term = trace(cov_a + cov_b - 2*(cov_proj.pow(0.5)))
    #print("first_term", first_term)
    #print("second term", second_term)
    return first_term


def random_attack(X, encoder, epsilon, p="inf"):

    delta = torch.randn_like(X).sign() * epsilon
    mu_a, cov_a = encoder(X)
    mu_b, cov_b = encoder(X + delta)
    loss = l2_wasserstein(mu_a, mu_b, cov_a, cov_b).mean()
    return delta, mu_b, cov_b, loss


def distortion_attack(X, encoder, decoder, num_steps, alpha, epsilon, p="inf"):
    """
    compute perturbation so as to increase the distortion
    :param X:
    :param encoder:
    :param decoder:
    :param num_steps:
    :param alpha:
    :param epsilon:
    :param p:
    :return:
    """

    delta = Variable(torch.randn_like(X) * 0.01, requires_grad=True)
    mu_b, cov_b = encoder(X + delta)
    Z = torch.bmm(cov_b, torch.randn(args.batch_size, mu_b.shape[1], 1).to(args.device))
    Z = Z.squeeze() + mu_b
    X_hat = decoder(Z)

    for n in range(num_steps):
        distortion = torch.pow(X_hat - X, 2).sum() / X.shape[0]
        distortion.backward(retain_graph=True)
        grad = delta.grad.detach()
        delta = get_projected_step(delta, grad, p, epsilon, alpha)
        # print("delta", delta)
        mu_b, cov_b = encoder(X + delta)
        Z = torch.bmm(cov_b, torch.randn(args.batch_size, mu_b.shape[1], 1).to(args.device))
        Z = Z.squeeze() + mu_b
        X_hat = decoder(Z)

    return delta, mu_b, cov_b, distortion


def rate_attack(X, encoder, num_steps, alpha, epsilon, p="inf"):
    """
    compute perturbation so as to decrease the KL divergence between the prior and posterior
    :param X:
    :param encoder:
    :param num_steps:
    :param epsilon:
    :param p:
    :return:
    """
    delta = Variable(torch.randn_like(X) * 0.01, requires_grad=True)
    mu_a, cov_a = encoder(X)
    mu_b, cov_b = encoder(X + delta)

    for n in range(num_steps):
        rate = kl_div(mu_b, cov_b).mean()
        rate.backward(retain_graph=True)
        grad = -delta.grad.detach()
        delta = get_projected_step(delta, grad, p, epsilon, alpha)
        #print("delta", delta)
        mu_b, cov_b = encoder(X + delta)

    return delta, mu_b, cov_b, rate


def l2_attack(X, encoder, num_steps=500, alpha=0.01, epsilon=0.03, p="inf", random_restarts=0):

    delta = Variable(torch.randn_like(X) * 0.01, requires_grad=True)
    mu_a, cov_a = encoder(X)
    mu_b, cov_b = encoder(X + delta)
    for n in range(num_steps):
        loss = l2_wasserstein(mu_a, mu_b, cov_a, cov_b).mean()
        #print("loss", loss)
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        #print("grad norm", grad[0].norm(p=2))
        delta = get_projected_step(delta, grad, p, epsilon, alpha)
        #print("delta", delta)
        mu_b, cov_b = encoder(X + delta)

    if p=='inf':
        delta = grad.sign() * epsilon
        mu_b, cov_b = encoder(X + delta)

    return delta, mu_b, cov_b, loss


def get_attack_stats(loader, encoder, decoder, beta, attack_type="l2_attack", num_samples=10, epsilon=0.05):

    clean_rate = AverageMeter()
    adv_rate = AverageMeter()

    clean_dist = AverageMeter()
    adv_dist = AverageMeter()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if attack_type == "l2_attack":
            delta, mu, cov, loss = l2_attack(X, encoder, num_steps=100, alpha=0.1, epsilon=epsilon, p="inf")

        elif attack_type == "rate_attack":
            delta, mu, cov, loss = rate_attack(X, encoder, num_steps=100, alpha=0.1, epsilon=epsilon, p="inf")

        elif attack_type == "distortion_attack":
            delta, mu, cov, loss = distortion_attack(X, encoder, decoder=decoder, num_steps=100,
                                                     alpha=0.1, epsilon=epsilon, p="inf")

        elif attack_type == "random_attack":
            delta, mu, cov, loss = random_attack(X, encoder, epsilon=epsilon, p="inf")

        Z = torch.bmm(cov.repeat(num_samples, 1, 1),
                      torch.randn(args.batch_size * num_samples, mu.shape[1], 1).to(args.device))

        Z = Z.squeeze() + mu.repeat(num_samples, 1)
        X_hat = decoder(Z)

        rate = 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mu, p=2, dim=1))
        distortion = torch.pow(X_hat - X.repeat(num_samples, 1, 1, 1), 2).sum(dim=(-1, -2, -3))
        adv_rate.update(rate, n=args.batch_size * num_samples)
        adv_dist.update(distortion, n=args.batch_size * num_samples)

        batch.set_description("Adv Rate {} Adv Distortion {}: ".format(adv_rate.avg, adv_dist.avg))

        X = X.repeat(num_samples, 1, 1, 1)
        mu, cov = encoder(X)
        Z = torch.bmm(cov, torch.randn(args.batch_size * num_samples, mu.shape[1], 1).to(args.device))
        Z = Z.squeeze() + mu
        X_hat = decoder(Z)

        rate = 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mu, p=2, dim=1))
        # note that distortion is calculated from one sample from the posterior
        distortion = torch.pow(X_hat - X, 2).sum(dim=(-1, -2, -3))

        clean_rate.update(rate, n=args.batch_size * num_samples)
        clean_dist.update(distortion, n=args.batch_size * num_samples)
        batch.set_description("Rate {} Distortion {}: ".format(clean_rate.avg, clean_dist.avg))

    # get histogram of rate and distortion for clean and adversarial
    # clean_rate.get_histogram(title="Rate", filename="rate_{}_{}.png".format(beta, attack_type))
    # adv_rate.get_histogram(title="Adv Rate", filename="adv_rate_{}_{}.png".format(beta, attack_type))
    # clean_dist.get_histogram(title="Distortion", filename="distortion_{}_{}.png".format(beta, attack_type))
    # adv_dist.get_histogram(title="Adv Distortion", filename="adv_distortion{}_{}.png".format(beta, attack_type))
    return clean_rate.avg, adv_rate.avg, clean_dist.avg, adv_dist.avg


def visualize_attack(test_loader, encoder, decoder):

    for X, y in test_loader:
        delta, mu, cov, loss = l2_attack(X, encoder, epsilon=0.05, p="inf", num_steps=500, alpha=0.01)
        X_adv = X + delta

        plt.imshow(X[0].permute(1, 2, 0))
        plt.show()

        plt.imshow(delta[0].permute(1, 2, 0).detach().numpy() * 50)
        plt.show()

        plt.imshow(X_adv[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        Z = torch.bmm(cov, torch.randn(args.batch_size, mu.shape[1], 1).to(args.device))
        Z = Z.squeeze() + mu
        X_hat = decoder(Z)

        plt.imshow(X_hat[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        KL = 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mu, p=2, dim=1))
        rate = KL.mean()
        # note that distortion is calculated from one sample from the posterior
        distortion = torch.pow(X_hat - X, 2).sum() / X.shape[0]

        print("Adv Rate {} Adv Distortion {}: ".format(rate, distortion))

        mu, cov = encoder(X)
        Z = torch.bmm(cov, torch.randn(args.batch_size, mu.shape[1], 1).to(args.device))
        Z = Z.squeeze() + mu
        X_hat = decoder(Z)

        plt.imshow(X_hat[0].permute(1, 2, 0).detach().numpy())
        plt.show()

        KL = 0.5 * (-log_diagonal_det(cov) - cov.shape[1] + trace(cov) + torch.norm(mu, p=2, dim=1))
        rate = KL.mean()
        # note that distortion is calculated from one sample from the posterior
        distortion = torch.pow(X_hat - X, 2).sum() / X.shape[0]

        print("Rate {} Distortion {}: ".format(rate, distortion))


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    train_loader, _ = celeb_loaders(args.batch_size)
    _, test_loader = celeb_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = ConvEncoder(num_channels=3, ndf=64, code_dim=args.code_size, dropout=args.dropout).to(args.device)
    decoder = DeconvDecoder(input_size=args.code_size).to(args.device)

    # TODO: sweep over beta values
    ckpt = torch.load(args.vae_ckpt, map_location=args.device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    #get_attack_stats(test_loader, encoder, decoder, beta=args.beta, attack_type="rate_attack", num_samples=10, epsilon=0.05)
    visualize_attack(test_loader, encoder, decoder)

