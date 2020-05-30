# a hack to ensure scripts search cwd
import sys

sys.path.append('../')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
import math
import os


def celeb_loaders(batch_size, shuffle_test=False):
    transform = transforms.Compose([transforms.Resize((64, 64), interpolation=2), transforms.ToTensor()])

    celeb_train = datasets.CelebA("data", split='train', target_type='attr', transform=transform,
                                download=True)

    celeb_test = datasets.CelebA("data", split="test", target_type="attr", transform=transform,
                                 download=True)

    train_loader = torch.utils.data.DataLoader(celeb_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(celeb_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def mnist_loaders(batch_size, shuffle_test=False):
    mnist_train = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def fashion_mnist_loaders(batch_size):
    mnist_train = datasets.MNIST("fashion_mnist", train=True,
                                 download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("fashion_mnist", train=False,
                                download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def replace_10_with_0(y):
    return y % 10


def svhn_loaders(batch_size):
    train = datasets.SVHN("./data", split='train', download=True, transform=transforms.ToTensor(),
                          target_transform=replace_10_with_0)
    test = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor(),
                         target_transform=replace_10_with_0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def har_loaders(batch_size):
    X_te = torch.from_numpy(np.loadtxt('data/UCI HAR Dataset/test/X_test.txt')).float()
    X_tr = torch.from_numpy(np.loadtxt('data/UCI HAR Dataset/train/X_train.txt')).float()
    y_te = torch.from_numpy(np.loadtxt('data/UCI HAR Dataset/test/y_test.txt') - 1).long()
    y_tr = torch.from_numpy(np.loadtxt('data/UCI HAR Dataset/train/y_train.txt') - 1).long()

    har_train = td.TensorDataset(X_tr, y_tr)
    har_test = td.TensorDataset(X_te, y_te)

    train_loader = torch.utils.data.DataLoader(har_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(har_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('./data', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def argparser(batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3,
              epsilon=0.1, starting_epsilon=None,
              proj=None,
              norm_train='l1', norm_test='l1',
              opt='sgd', momentum=0.9, weight_decay=5e-4):
    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=10)

    # projection settings
    parser.add_argument('--proj', type=int, default=proj)
    parser.add_argument('--norm_train', default=norm_train)
    parser.add_argument('--norm_test', default=norm_test)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon
    if args.prefix:
        if args.model is not None:
            args.prefix += '_' + args.model

        if args.method is not None:
            args.prefix += '_' + args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval',
                  'method', 'model', 'cuda_ids', 'load', 'real_time',
                  'test_batch_size']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length',
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # Ignore these parameters for filename since we never change them
        banned += ['momentum', 'weight_decay']

        if args.cascade == 1:
            banned += ['cascade']

        # if not using a model that uses model_factor,
        # ignore model_factor
        if args.model not in ['wide', 'deep']:
            banned += ['model_factor']

        # if args.model != 'resnet':
        banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)):
            if arg not in banned and getattr(args, arg) is not None:
                args.prefix += '_' + arg + '_' + str(getattr(args, arg))

        if args.schedule_length > args.epochs:
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else:
        args.prefix = 'temporary'

    if args.cuda_ids is not None:
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    return args
