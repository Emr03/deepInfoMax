import torch
import torch.nn as nn
from torch.autograd import Variable
from attacks.vae_attacks import l2_wasserstein


def cw_vae_encoder_attack(X_src, X_tgt, encoder, num_steps=5000, alpha=0.001, c=10., p=float("inf"),
                      dim=(-1, -2, -3)):
    """
    implements carlini-wagner attack for wasserstein distance between posterior of source and that of target image
    :param X_src:
    :param X_tgt:
    :param encoder:
    :param num_steps:
    :param alpha:
    :param c:
    :param epsilon:
    :param p:
    :param random_restarts:
    :return:
    """
    # stable computation of tanh inverse, to intitialize w s.t delta is close to 0
    X_src = torch.clamp(X_src, min=0.01, max=1 - 0.01)
    y = 2 * (X_src) - 1
    w = torch.tensor(0.5 * (torch.log1p(y) - torch.log1p(-y)), requires_grad=True)

    # implementation of block constraint
    delta = 0.5 * (torch.tanh(w) + 1) - X_src
    opt = torch.optim.Adam(params=[w], lr=alpha)

    mu_src, cov_src = encoder(X_src)
    mu_b, cov_b = encoder(X_src + delta)
    mu_tgt, cov_tgt = encoder(X_tgt)

    for n in range(num_steps):
        opt.zero_grad()
        main_loss = l2_wasserstein(mu_tgt, mu_b, cov_tgt, cov_b)
        delta_penalty = torch.norm(delta, p=p, dim=dim)
        print(n, " delta norm ", delta_penalty.mean(), "main_loss: ", main_loss.mean())
        loss = main_loss.mean() + c * delta_penalty.mean()
        loss.backward(retain_graph=True)
        opt.step()
        delta = 0.5 * (torch.tanh(w) + 1) - X_src
        #print("delta", delta)
        mu_b, cov_b = encoder(X_src + delta)

    return delta, mu_b, cov_b, main_loss


def cw_infomax_encoder_attack(X_src, X_tgt, encoder, num_steps=5000, alpha=0.001, c=10., p=float("inf"),
                      dim=(-1, -2, -3)):

    # stable computation of tanh inverse, to intitialize w s.t delta is close to 0
    X_src = torch.clamp(X_src, min=0.01, max=1 - 0.01)
    y = 2 * (X_src) - 1
    w = torch.tensor(0.5 * (torch.log1p(y) - torch.log1p(-y)), requires_grad=True)

    # implementation of block constraint
    delta = 0.5 * (torch.tanh(w) + 1) - X_src
    opt = torch.optim.Adam(params=[w], lr=alpha)

    _, _, Z_src = encoder(X_src)
    _, _, Z_b = encoder(X_src + delta)
    _, _, Z_tgt = encoder(X_tgt)

    for n in range(num_steps):
        opt.zero_grad()
        main_loss = torch.norm(Z_b - Z_tgt, p=2, dim=-1).mean()
        delta_penalty = torch.norm(delta, p=p, dim=dim)
        print(n, " delta norm ", delta_penalty.mean(), "main_loss: ", main_loss.mean())
        loss = main_loss.mean() + c * delta_penalty.mean()
        loss.backward(retain_graph=True)
        opt.step()
        delta = 0.5 * (torch.tanh(w) + 1) - X_src
        # print("delta", delta)
        _, _, Z_b = encoder(X_src + delta)

    return delta, Z_b, main_loss

