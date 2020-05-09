"""
compute adversarial example X such that MI(X, E) is lowered
        # 1. encoder_attack: compute delta s.t ||Enc(X + delta) - Enc(X)|| is maximized


        # 2. critic_attack: compute delta s.t score of critic for (Enc(X) + delta, X) is minimized


        # 3. can adversarially construct X E pairs by modifying a randomly selected X s.t X + delta, E has a high score
"""
from attacks.gradient_untargeted import pgd, fgsm
from attacks.gradient_untargeted import get_projected_step
import torch
import torch.nn as nn

def encoder_attack(X, encoder, num_steps, epsilon, alpha, random_restart=True):
    """
    perform PGD encoder attack
    :param X:
    :param encoder:
    :return:
    """

    if random_restart:
        delta = torch.randn_like(X, requires_grad=True)

    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # print("grad", delta.requires_grad)
    _, Z = encoder(X)
    Z = Z.detach()
    for n in range(num_steps):
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        loss = torch.norm(encoder(X + delta) - Z, p=2, dim=-1)
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        # print("grad", grad)
        delta = get_projected_step(delta, grad, 2, epsilon, alpha)

    X_adv = X + delta
    E_adv = encoder(X_adv)
    return X_adv, E_adv

def critic_attack(E, critic, num_steps, random_restart=True):

    # critic_attack: compute delta s.t score of critic for (Enc(X) + delta, X) is minimized
    if random_restart:
        delta = torch.randn_like(E, requires_grad=True)

    else:
        delta = torch.zeros_like(E, requires_grad=True)

    for n in range(num_steps):
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        loss = torch.norm(encoder(X + delta) - Z, p=2, dim=-1)
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        # print("grad", grad)
        delta = get_projected_step(delta, grad, 2, epsilon, alpha)

    E_adv = E + delta
    return X_adv, E_adv




