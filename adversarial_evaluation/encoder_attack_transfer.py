import torch
import torch.nn as nn
import os
import glob
from attacks.gradient_targeted import cw_infomax_encoder_attack, cw_vae_encoder_attack
from attacks.vae_attacks import l2_wasserstein
from utils.argparser import argparser
from utils import data_loaders
from models.decoders import *
from utils import get_config
import random
from tqdm import tqdm
import numpy as np
from utils.train_eval import AverageMeter


def infomax_transfer(src_encoder, tgt_encoder, src_decoder, tgt_decoder, loader, log, gpu):

    main_loss = AverageMeter()
    transfer_loss = AverageMeter()
    decoder_loss = AverageMeter()
    transfer_decoder_loss = AverageMeter()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for X, y in batch:
        if gpu:
            X = X.cuda()

        batch_size = X.shape[0]
        X_s = X[0:batch_size // 2]
        X_t = X[batch_size // 2:]
        delta, Z_b, loss = cw_infomax_encoder_attack(X_s, X_t, encoder=src_encoder,
                                                     num_steps=2000, alpha=0.001, c=0.1, p=2)

        # evaluate decoding wrt target image
        _, _, Z_tgt = src_encoder(X_t)
        X_hat_tgt = src_decoder(Z_tgt)
        X_hat_adv = src_decoder(Z_b)

        recon_loss = torch.norm(X_hat_tgt - X_hat_adv, p=2, dim=(-3, -2, -1)).mean()
        decoder_loss.update(recon_loss)
        main_loss.update(loss)

        # compute transfer losses
        _, _, Z = tgt_encoder(X_s)
        _, _, Z_adv = tgt_encoder(X_s + delta)
        _, _, Z_tgt = tgt_encoder(X_t)

        X_hat_tgt = tgt_decoder(Z_tgt)
        loss = torch.norm(Z_tgt - Z_adv, dim=-1, p=2).mean()
        X_hat_adv = tgt_decoder(Z_adv)
        recon_loss = torch.norm(X_hat_tgt - X_hat_adv, p=2, dim=(-3, -2, -1)).mean()
        transfer_loss.update(loss)
        transfer_decoder_loss.update(recon_loss)


    print("Encoder Loss: {}\t "
          "Transfer Encoder Loss: {}\t"
          "Decoder Matching Loss {}\t"
          "Transfer Decoder Matching Loss ".format(main_loss.avg, transfer_loss.avg,
                                                   decoder_loss.avg, transfer_decoder_loss.avg), file=log)

    log.flush()


def vae_transfer(src_vae, tgt_vae, loader, log, gpu):

    main_loss = AverageMeter()
    transfer_loss = AverageMeter()
    decoder_loss = AverageMeter()
    transfer_decoder_loss = AverageMeter()

    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for X, y in batch:
        if gpu:
            X = X.cuda()

        batch_size = X.shape[0]
        X_s = X[0:batch_size // 2]
        X_t = X[batch_size // 2:]
        delta, Z_b, loss = cw_vae_encoder_attack(X_s, X_t, encoder=src_encoder,
                                                     p=2, num_steps=2500, c=1.5, alpha=0.001)

        # evaluate decoding wrt target image
        _, _, _, X_hat_tgt = src_vae(X_t)
        _, _, Z_b, X_hat_adv = src_vae(X_s + delta)
        decoder_loss.update(torch.norm(X_hat_adv - X_hat_tgt, p=2, dim=(-3, -2, -1)).mean())
        main_loss.update(loss)

        mu_tgt, cov_tgt = tgt_vae.encoder(X_t)
        Z_tgt = torch.bmm(cov_tgt, torch.randn(batch_size, mu_tgt.shape[1], 1).to(X.device))
        Z_tgt = Z_tgt.squeeze() + mu_tgt.squeeze()

        mu_adv, cov_adv = tgt_vae.encoder(X_s + delta)
        Z_adv = torch.bmm(cov_adv, torch.randn(batch_size, mu_adv.shape[1], 1).to(X.device))
        Z_adv = Z_adv.squeeze() + mu_adv.squeeze()

        X_hat_tgt = tgt_vae.decoder(Z_tgt)
        X_hat_adv = tgt_vae.decoder(Z_adv)
        loss = l2_wasserstein(mu_tgt, mu_adv, cov_tgt, cov_adv)
        transfer_decoder_loss.update(torch.norm(X_hat_adv - X_hat_tgt, p=2, dim=(-3, -2, -1)).mean())
        transfer_loss.update(loss)

    print("Encoder Loss: {}\t "
          "Transfer Encoder Loss: {}\t"
          "Decoder Matching Loss {}\t"
          "Transfer Decoder Matching Loss ".format(main_loss.avg, transfer_loss.avg,
                                                   decoder_loss.avg, transfer_decoder_loss.avg), file=log)

    log.flush()


if __name__ == "__main__":

    args = argparser()
    print("saving file to {}".format(args.prefix))

    # create workspace
    workspace_dir = "experiments/{}".format(args.prefix)
    if not os.path.isdir(workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)

    log = open("{}/transfer.log".format(workspace_dir), "a")
    input_size, ndf, num_channels, train_loader, test_loader = get_config(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)

    infomax_encoder_ckpts = []
    infomax_decoder_ckpts = []
    vae_ckpts = []
    for dir in glob.iglob("experiments/encoders/{}".format(args.data)):
        encoder_ckpt = "{}/{}_checkpoint.pth".format(dir, dir)
        decoder_ckpt = "experiments/decoders/decoder_{}/decoder_{}_checkpoint.pth".format(dir, dir)
        infomax_encoder_ckpts.append(encoder_ckpt)
        infomax_decoder_ckpts.append(decoder_ckpt)

    for dir in glob.iglob("experiments/vaes/{}/".format(args.data)):
        vae_ckpt = "{}/{}_checkpoint.pth".format(dir, dir)

    ### infomax to infomax transfer ###

    # create source and target encoders and decoders
    # TODO: device coordination
    src_encoder = GlobalEncoder(ndf=ndf, num_channels=num_channels,
                                output_size=args.code_size, input_size=input_size)
    src_encoder = src_encoder.to(args.device)
    src_encoder.eval()

    src_decoder = DeconvDecoder(input_size=src_encoder.output_size, output_size=input_size,
                                output_channels=num_channels, ndf=ndf)
    src_decoder = src_decoder.to(args.device)
    src_decoder.eval()

    tgt_encoder = GlobalEncoder(ndf=ndf, num_channels=num_channels,
                                output_size=args.code_size, input_size=input_size)
    tgt_encoder = src_encoder.to(args.device)
    tgt_encoder.eval()

    tgt_decoder = DeconvDecoder(input_size=src_encoder.output_size, output_size=input_size,
                                output_channels=num_channels, ndf=ndf)
    tgt_decoder = src_decoder.to(args.device)
    tgt_decoder.eval()

    for i, ckpt_i in enumerate(infomax_encoder_ckpts):
        for j, ckpt_j in enumerate(infomax_encoder_ckpts):

            if ckpt_i == ckpt_j:
                continue

            src_encoder.load_state_dict(torch.load(ckpt_i, map_location=args.device)["encoder_state_dict"])
            src_decoder.load_state_dict(torch.load(ckpt_i, map_location=args.device)["encoder_state_dict"])

            tgt_encoder.load_state_dict(torch.load(ckpt_j, map_location=args.device)["encoder_state_dict"])
            tgt_decoder.load_state_dict(torch.load(ckpt_j, map_location=args.device)["encoder_state_dict"])

            print("Source Model: {}\t Target Model: {}\t".format(ckpt_i, ckpt_j), file=log)
            log.flush()

            infomax_transfer(src_encoder, tgt_encoder, src_decoder, tgt_decoder, test_loader, log, args.gpu)

        # TODO vae - vae attack transfer

        # TODO infomax - vae attack transfer