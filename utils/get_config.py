"""
Return encoder and decoder hyperparams as well as data loaders based on args.data
"""

import utils.data_loaders
from models.encoders import *
from models.decoders import *
from utils import data_loaders

def get_config(args):

    if args.data == "cifar10":
        train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
        _, test_loader = data_loaders.cifar_loaders(args.batch_size)
        input_size = 32
        num_channels=3
        ndf=64

    elif args.data == "celeb":
        train_loader, _ = data_loaders.celeb_loaders(args.batch_size)
        _, test_loader = data_loaders.celeb_loaders(args.batch_size)
        input_size = 64
        num_channels=3
        ndf=64

    elif args.data == "mnist":
        train_loader, _ = data_loaders.mnist_loaders(args.batch_size)
        _, test_loader = data_loaders.mnist_loaders(args.batch_size)
        input_size = 28
        num_channels=1
        ndf=16

    return input_size, ndf, num_channels, train_loader, test_loader

