import torch
import torch.nn as nn
from models.classifier import *
from models.encoders import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils.argparser import argparser
from utils import data_loaders
from attacks.gradient_untargeted import pgd, fgsm
import os
import random

def tsne(X_list, model=None, cmap_name='tab10', filename=None):
    """

    :param X_list: list of numpy arrays containing representations for a given class
    :return:
    """
    # concatenate all representations to fit t-SNE
    X = np.concatenate(X_list, axis=0)
    if not model:
        model = TSNE(n_components=2)

    X_emb = model.fit_transform(X)
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots()

    start = 0
    for i, l in enumerate(X_list):
        N = len(l)
        end = start + N
        color = cmap(i/len(X_list))
        print(color)
        ax.scatter(X_emb[start:end, 0], X_emb[start:end, 1], color=color)
        start = end

    if not filename:
        plt.show()

    else:
        plt.savefig(filename)

    return model


def test_tsne(dim=50):

    X_list = []
    for i in range(10):
        mean = np.random.rand(1, 50) * 4
        X_list.append(np.random.randn(75, 50) * 2 + mean)

    tsne(X_list)


def sort_by_label(Z, y, num_classes=10):

    Z_list = []
    y_list = []
    for c in range(num_classes):
        mask = np.where(y == c)[0]
        Z_list.append(Z[mask, :])
        y_list.append(y[mask])

    return Z_list, y_list

def test_sort_by_label():
    Z = np.random.rand(75, 5)
    y = np.random.randint(low=0, high=10, size=(75,))
    z_list, y_list = sort_by_label(Z, y)
    print(z_list, y_list)


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace, to save t-SNE plots
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)

    _, test_loader = data_loaders.cifar_loaders(batch_size=args.batch_size)

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
    classifier.load_state_dict(torch.load(args.classifier_ckpt)["classifier_state_dict"]) 
    classifier.to(args.device)

    Z = []
    pred = []
    Y = []
    Z_adv = []
    pred_adv = []

    for X, y in test_loader:
        if args.gpu:
            X, y = X.cuda(), y.cuda()
            with torch.no_grad():
                z, logits = classifier(X, intermediate=True)
                Z.append(z.cpu().detach().numpy())
                pred.append(logits.max(-1)[1].cpu().detach().numpy())
                Y.append(y.cpu().detach().numpy())

                X_adv, delta, out, out_adv = pgd(model=classifier, X=X, y=y, epsilon=args.epsilon,
                                                 alpha=args.alpha, num_steps=args.num_steps, p='inf')

                z, logits = classifier(X_adv, intermediate=True)
                Z_adv.append(z.cpu().detach().numpy())
                pred_adv.append(out_adv.cpu().detach().numpy())

    Z = np.concatenate(Z, axis=0)
    pred = np.concatenate(pred, axis=0)
    Y = np.concatenate(Y, axis=0)

    # make visualization for ground truth labels
    z_list, y_list = sort_by_label(Z, Y, num_classes=10)
    tsne_model = tsne(z_list, filename="{}/label_tsne.png".format(workspace_dir))

    # make visualization for predicted labels
    z_list, pred_list = sort_by_label(Z, pred, num_classes=10)
    tsne(z_list, model=tsne_model, filename="{}/pred_tsne.png".format(workspace_dir))

    # make visualization for adversarial inputs
    # sort by ground truth
    z_list, y_list = sort_by_label(Z_adv, Y, num_classes=10)
    tsne(z_list, tsne_model, filename="{}/adv_gt_tsne.png".format(workspace_dir))

    # sort by prediction
    z_list, pred_list = sort_by_label(Z_adv, pred_adv, num_classes=10)
    tsne(z_list, tsne_model, filename="{}/adv_pred_tsne.png".format(workspace_dir))












