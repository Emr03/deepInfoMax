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
    perform PGD encoder attack, optimize delta so as to maximize the difference between encoder outputs
    :param X:
    :param encoder:
    :return:
    """

    if random_restart:
        delta = torch.randn_like(X, requires_grad=True)

    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # print("grad", delta.requires_grad)
    _, _, Z = encoder(X)
    Z = Z.detach()
    for n in range(num_steps):
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        _, _, E = encoder(X + delta)
        # to maximize norm, minimize -L2 norm 
        loss = -torch.norm(E - Z, p=2, dim=-1)
        loss.mean().backward(retain_graph=True)
        grad = delta.grad.detach()
        # print("grad", grad)
        delta = get_projected_step(delta, grad, "inf", epsilon, alpha)

    X_adv = X + delta
    _, _, E_adv = encoder(X_adv)
    return X_adv, E_adv, -loss.mean(), -loss.min()


def critic_attack(E, critic, num_steps, random_restart=True):

    pass

def source2target(X_s, X_t, encoder, epsilon, step_size, max_steps=500, c=1.0, random_restart=True):

    _, _, Z_s = encoder(X_s)
    _, _, Z_t = encoder(X_t)

    if random_restart:
        w = torch.randn_like(X_s, requires_grad=True)

    else:
        w = torch.zeros_like(X_s, requires_grad=True)
     
    # implementation of block constraint
    delta = 0.5 * (torch.tanh(w) + 1) - X_s
    for n in range(max_steps):
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        _, _, Z_s = encoder(X_s + delta)
        z_norm = torch.norm(Z_s - Z_t, p=2, dim=-1) 
        delta_norm = torch.norm(delta, p=2, dim=(-1, -2, -3))
        loss = z_norm + c * delta_norm
        loss.mean().backward(retain_graph=True)
        grad = w.grad.detach()
        # print("grad", grad)
        print(z_norm.mean(), delta_norm.mean())
        #delta = get_projected_step(delta, grad, 2, epsilon, step_size)

    X_adv = X_s + delta
    return X_adv, Z_s, z_norm.mean(), z_norm.min()



