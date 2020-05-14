"""
Implements commonly used gradient-based white-box untargeted attacks for different norm constraints
Each function takes as input the batch of inputs, labels, model, norm constraint,
norm type as well as additional keyword arguments
Each function returns the perturbed input, the perturbation vector, the old prediction and the new prediction
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


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


def fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    #loss = loss_fn(model(X + delta), y)
    loss.backward()
    perturbation = epsilon * delta.grad.detach().sign()

    X_adv = X + perturbation
    out = torch.argmax(model(X), dim=-1)
    out_adv = torch.argmax(model(X_adv), dim=-1)
    return X_adv, perturbation, out, out_adv


def pgd(model, X, y, epsilon, alpha, num_steps, random_restart=False, p=2):
    """

    :param model:
    :param X:
    :param y:
    :param epsilon: max p-norm of perturbation
    :param alpha: step size
    :param num_steps: num of ascent steps
    :param random_restart: whether to perturb X before beginning pgd
    :param p: oder of norm of attack, either 2 or "inf" or 0
    :return:
    """

    if random_restart:
        delta = torch.randn_like(X, requires_grad=True)

    else:
        delta = torch.zeros_like(X, requires_grad=True)

    # print("grad", delta.requires_grad)
    for n in range(num_steps):
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        # print("grad", grad)
        delta = get_projected_step(delta, grad, p, epsilon, alpha)

    X_adv = X + delta
    out = torch.argmax(model(X), dim=-1)
    out_adv = torch.argmax(model(X_adv), dim=-1)
    return X_adv, delta, out, out_adv

