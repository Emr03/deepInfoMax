import torch
import torch.nn as nn
import numpy as np

def generate(batch_size, R, d):

    # sample x from N(0, I) and normalize
    x = torch.randn(size=(batch_size, d))
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).repeat(1, d)
    x = x / norm

    # sample binary labels and scale accordingly
    y = torch.randint(size=(batch_size, 1), low=0, high=2).float()
    x = x + (R - 1) * y * x # if y = 0 x, else R * x
    return x, y

def get_model(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 1000),
                         nn.ReLU(),
                         nn.BatchNorm1d(1000),
                         nn.Linear(1000, 1000),
                         nn.ReLU(),
                         nn.BatchNorm1d(1000),
                         nn.Linear(1000, 1))

def train(input_dim=500, steps=1000000, batch_size=50, R=1.3):

    model = get_model(input_dim=input_dim).cuda()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(lr=0.0001, params=model.parameters())
    for t in range(steps):
        X, Y = generate(batch_size, R, input_dim)
        X = X.cuda()
        Y = Y.cuda()
        loss = loss_fn(model(input=X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % 1000 == 0:
            print("Step {}: loss {}".format(t, loss))

    torch.save(model.state_dict(), f="adv_spheres_model.pth")

if __name__ == "__main__":
    train()

