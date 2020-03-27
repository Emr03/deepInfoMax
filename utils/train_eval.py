import time
import torch.nn as nn
from tqdm import tqdm

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


def train_dim(loader, model, enc_opt, T_opt, epoch, log, verbose, gpu):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    batch = tqdm(loader, total=len(loader) // loader.batch_size)

    for i, (X,y) in enumerate(batch):
        if gpu:
            X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        loss = model(X)

        enc_opt.zero_grad()
        T_opt.zero_grad()
        loss.backward()
        enc_opt.step()
        T_opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(loss.item(), X.size(0))

        batch.set_description("Epoch {} Loss {} ".format(epoch, losses.avg))
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses), file=log)
        log.flush()

    return losses.avg

def train_classifier(loader, model, opt, epoch, log, verbose, gpu):

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

    return losses.avg

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
