import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from adversarial_spheres.main import generate, get_model
from matplotlib import pyplot as plt

def evaluate(model, batch_size=1000, d=500, R=1.3):

    avg_err = 0
    model = model.eval()
    for step in range(2000):
        X, Y = generate(batch_size, R, d)
        out = torch.round(F.sigmoid(model(X)))
        err = (out.data != Y.data).sum()
        avg_err += err
        print("Step: {}, Error {}".format(step, err))

    print("Avg Error", avg_err / 2000)

def manifold_attack(model, num_steps=1000, step_size=0.01, R=1.3, d=500):

    model = model.eval()
    X, Y = generate(batch_size=1, R=R, d=d)
    out = torch.round(F.sigmoid(model(X)))
    noise = torch.randn_like(X) * 0.01
    # for the projection step, project back on the sphere by normalizing x
    X = torch.autograd.Variable(X, requires_grad=True)
    for n in range(num_steps):
        #print("X", X)
        # x = torch.autograd.Variable(X.data, requires_grad=True)
        loss = nn.BCEWithLogitsLoss()(model(X), Y)
        loss.backward()
        print(model(X), Y, loss)
        grad = X.grad.detach()
        grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True).repeat(1, d)

        #print("grad", grad)
        X = X + step_size * grad / grad_norm
        #print(X)
        norm = torch.norm(X, p=2, dim=-1, keepdim=True).repeat(1, d)
        X = X / norm
        X = X + (R - 1) * Y * X
        X = torch.autograd.Variable(X, requires_grad=True)

    out_adv = torch.round(F.sigmoid(model(X)))
    return X, out, out_adv

def visualize(model, v1, v2, N=1000):

    # N x 500
    X_p = [torch.randn(size=(N, 1)) * 500, torch.randn(size=(N, 1)) * 500]
    X_1 = v1.repeat(N, 1) * X_p[0]
    X_2 = v2.repeat(N, 1) * X_p[1]

    X = X_1 + X_2
    n = torch.norm(X, p=2, dim=-1, keepdim=True).repeat(1, 500)
    X = X / n
    pred = F.sigmoid(model(X))

    V = torch.cat([v1, v2], dim=0).transpose(1, 0)
    print(V.shape)
    X_p = torch.mm(X, V)
    print(X_p.shape)

    U = np.linspace(start=0, stop=2*np.pi, num=100)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    pts = (np.cos(U), np.sin(U))
    ax.plot(pts[0], pts[1], c='black')

    pts = (1.3 * np.cos(U), 1.3 * np.sin(U))
    ax.plot(pts[0], pts[1], c='orange')

    # select inputs with prediction Y = 0
    y_0 = torch.where(pred < 0.5)[0]
    ax.scatter(X_p[:, 0][y_0], X_p[:, 1][y_0], c='green')

    y_1 = torch.where(pred >= 0.5)[0]
    ax.scatter(X_p[:, 0][y_1], X_p[:, 1][y_1], c='yellow')
    plt.show()

if __name__ == "__main__":

    torch.random.manual_seed(498)
    model = get_model(input_dim=500)
    model.load_state_dict(torch.load("../adv_spheres_model_2.pth", map_location=torch.device('cpu')))

    evaluate(model)
    X, out, out_adv = manifold_attack(model)
    print((out.data != out_adv).sum() / X.shape[0])

    # v1 = torch.randn(size=(1, 500))
    # n = torch.norm(v1, p=2)
    # v1 = v1 / n
    #
    # v2 = torch.randn(size=(1, 500))
    # n = torch.norm(v2, p=2)
    # v2 = v2 / n
    #
    # visualize(model, v1, v2)



