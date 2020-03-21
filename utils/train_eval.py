import time

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
    for i, (X,y) in enumerate(loader):
        print(X.shape, i, len(loader))
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

        print(epoch, i, loss.item())
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses), file=log)
        log.flush()

    return losses.avg
