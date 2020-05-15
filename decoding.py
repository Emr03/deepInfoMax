import torch
import torch.nn as nn
from utils.data_loaders import *
from models.mi_estimation import *
from models.encoders import *
from models.decoder import *
from utils.argparser import argparser
from utils import data_loaders
from utils import train_eval
import random
import numpy as np
import json

if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    # save arguments as json file
    # json.dump(obj=args, separators="\t", indent=4, fp="{}_args".format(workspace_dir))

    train_log = open("{}/train.log".format(workspace_dir), "w")
    test_log = open("{}/test.log".format(workspace_dir), "w")

    train_loader, _ = data_loaders.cifar_loaders(args.batch_size)
    _, test_loader = data_loaders.cifar_loaders(args.batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    encoder = GlobalEncoder(stride=args.encoder_stride)
    # load encoder from checkpoint
    encoder.load_state_dict(torch.load(args.encoder_ckpt)["encoder_state_dict"])
    encoder = encoder.to(args.device)

    decoder = DecoderY(input_size=encoder.output_size)
    decoder = decoder.to(args.device)
    opt = optim.Adam(decoder.parameters(), lr=args.lr)
    e = 0

    if args.decoder_ckpt:
        ckpt = torch.load(args.decoder_ckpt)
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        opt.load_state_dict(ckpt["opt"])
        e = ckpt["epoch"]

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[100, 300, 500, 700, 900], gamma=0.5)
    # if num of visible devices > 1, use DataParallel wrapper
    while e < args.epochs:
        loss = train_eval.train_decoder(train_loader, encoder, decoder, opt, e,
                                           train_log, verbose=args.verbose, gpu=args.gpu)
        e += 1
        #scheduler.step()

        #train_eval.eval_decoder(test_loader, decoder, e, test_log, verbose=args.verbose, gpu=args.gpu)
        torch.save({
            'decoder_state_dict': decoder.state_dict(),
            'epoch': e,
            'opt': opt.state_dict(),
            'loss': loss,
        }, workspace_dir + "/" + args.prefix + "_checkpoint.pth")






