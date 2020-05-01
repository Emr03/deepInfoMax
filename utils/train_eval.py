import time
import torch.nn as nn
from tqdm import tqdm
from utils.ms_ssim import ms_ssim
from attacks.gradient_untargeted import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_dim(loader, model, enc_opt, T_opt, epoch, log, verbose, gpu, prior_matching=None, D_opt=None, beta=1, gamma=0.1):
    """

    :param loader: train data loader
    :param model: DIM model includes encoder and mi estimator
    :param enc_opt: optimizer for encoder params
    :param T_opt: optimizer for mi estimator params
    :param epoch:
    :param log: log file
    :param verbose:
    :param gpu:
    :param prior_matching: module for prior matching discriminator, default None
    :param D_opt: optimizer for prior matching discriminator, default None
    :return:
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    dim_losses = AverageMeter()
    prior_losses = AverageMeter()
    model.train()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)

    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        dim_loss, E = model(X)
        dim_loss = dim_loss.mean()
        
        T_opt.zero_grad()
        enc_opt.zero_grad()

        if prior_matching:
            D_opt.zero_grad()
            prior_matching_loss = prior_matching(E).mean()
            
            d_loss = prior_matching_loss

            d_loss.backward(retain_graph=True)
            D_opt.step()

            enc_opt.zero_grad()
            e_loss = dim_loss * beta - prior_matching_loss * gamma
            e_loss.backward()
            enc_opt.step()
            T_opt.step()
            prior_losses.update(prior_matching_loss.item(), X.size(0))

        else:
            dim_loss.backward()
            enc_opt.step()
            T_opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        dim_losses.update(dim_loss.item(), X.size(0))
        
        batch.set_description("Epoch {} DIM Loss {} Prior Loss {}".format(epoch, dim_losses.avg, prior_losses.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'DIM Loss {dim_loss.val:.4f} ({dim_loss.avg:.4f})\t'
                  'Prior Loss {prior_loss.val:.4f} ({prior_loss.avg:.4f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, dim_loss=dim_losses, prior_loss=prior_losses), file=log)
        log.flush()

    return dim_losses.avg

def train_classifier(loader, model, opt,  epoch, log, verbose, gpu):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    model.train()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        # get output logits
        output = model(X)
        loss = nn.CrossEntropyLoss()(input=output, target=y)
        err = (output.data.max(1)[1] != y).float().sum() / X.size(0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(loss.item(), X.size(0))
        errors.update(err, X.size(0))

        batch.set_description("Epoch {} Loss {} Err {} ".format(epoch, losses.avg, errors.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, errors=errors), file=log)

        log.flush()

    return losses.avg

def eval_classifier(loader, model, epoch, log, verbose, gpu):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    model.train()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        # get output logits
        output = model(X)
        loss = nn.CrossEntropyLoss()(input=output, target=y)
        err = (output.data.max(1)[1] != y).float().sum() / X.size(0)

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), X.size(0))
        errors.update(err, X.size(0))

        batch.set_description("Epoch {} Test Loss {} Test Err {} ".format(epoch, losses.avg, errors.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, errors=errors), file=log)

        log.flush()

    return errors.avg

def train_decoder(loader, encoder, decoder, opt, epoch, log, verbose, gpu):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    decoder.train()
    encoder.eval()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        C, E = encoder(X)
        R = decoder(E)
        loss = nn.MSELoss()(input=R, target=X)

        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(loss.item(), X.size(0))

        batch.set_description("Epoch {} Loss {}".format(epoch, losses.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses), file=log)

        log.flush()

    return losses.avg

def eval_decoder(loader, encoder, decoder, log, verbose, gpu):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ms_ssim_val = AverageMeter()
    decoder.eval()
    encoder.eval()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        C, E = encoder(X)
        if len(E.shape) == 1:
            E = E.unsqueeze(0)
        R = decoder(E)
        sim = ms_ssim(X, R, size_average=False)

        batch_time.update(time.time() - end)
        end = time.time()
        ms_ssim_val.update(sim.item(), X.size(0))

        if verbose and i % verbose == 0:
            print('Batch: [{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, batch_time=batch_time,
                data_time=data_time, loss=ms_ssim_val), file=log)

    return ms_ssim_val.avg


def train_classifier_adversarial(loader, model, opt, epoch, log, verbose, gpu, args, gamma=0.9):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    clean_losses = AverageMeter()
    clean_errors = AverageMeter()

    adv_losses = AverageMeter()
    adv_errors = AverageMeter()

    model.train()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()

        out = model(X)
        ce_clean = nn.CrossEntropyLoss()(out, y)
        err_clean = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # adv samples
        if args.attack == "pgd":
            X_adv, delta, out, out_adv = pgd(model=model, X=X, y=y, epsilon=args.epsilon,
                                             alpha=args.alpha, num_steps=args.num_steps, p="inf")

        elif args.attack == "fgsm":
            X_adv, delta, out, out_adv = fgsm(model=model, X=X, y=y, epsilon=args.epsilon)

        X = Variable(X + delta)
        out = model(Variable(X))
        ce_adv = nn.CrossEntropyLoss()(out, Variable(y))
        err_adv = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        ce_total = (1-gamma)*ce_clean + gamma*ce_adv
        opt.zero_grad()
        ce_total.backward()
        opt.step()
        
        # measure accuracy and record loss
        clean_losses.update(ce_clean.item(), X.size(0))
        clean_errors.update(err_clean, X.size(0))
        adv_losses.update(ce_adv.item(), X.size(0))
        adv_errors.update(err_adv, X.size(0))
        
        batch.set_description("clean loss: {}, "
                              "adv loss: {}, "
                              "clean_err: {}, "
                              "adv_err: {}, ".format(ce_clean.item(), ce_adv.item(), err_clean, err_adv))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' Epoch {epoch}:\t Clean Error {clean_error.avg:.3f}\t'
          ' Adv Error {adv_errors.avg:.3f}\t'
          .format(epoch=epoch, clean_error=clean_errors, adv_errors=adv_errors), file=log)
    
    log.flush()
    return clean_errors.avg

def ndm_train(loader, model, opt, epoch, log, verbose, gpu):

    model.train()
    ndm = AverageMeter()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()

        ndm_loss=model(X)
        ndm.update(ndm_loss)

        opt.zero_grad()
        ndm_loss.backward()
        opt.step()

        batch.set_description("Epoch {} Loss {}".format(epoch, ndm.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader),
                loss=ndm), file=log)

        log.flush()

    return ndm.avg

def ndm_eval(loader, model, log, gpu):

    model.eval()
    ndm = AverageMeter()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)
    for i, (X, y) in enumerate(batch):
        if gpu:
            X, y = X.cuda(), y.cuda()

        ndm_loss = model(X)
        ndm.update(ndm_loss)

        batch.set_description("iter {} Loss {}".format(i, ndm.avg))
        print('iter: [{0}/{1}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            i, len(loader),
            loss=ndm), file=log)

        log.flush()
