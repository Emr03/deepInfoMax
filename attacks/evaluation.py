from torch.autograd import Variable
import time
import torch.nn as nn
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


def evaluate_adversarial(args, model, loader):
    """
    Only implements adversarial attacks on classification for now
    :param opt:
    :param model:
    :param loader:
    :return:
    """
    batch_time = AverageMeter()
    clean_losses = AverageMeter()
    clean_errors = AverageMeter()
    adv_losses = AverageMeter()
    adv_errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X, y) in enumerate(loader):

        if args.gpu:
            X, y = X.cuda(), y.cuda()

        out = model(Variable(X))
        ce_clean = nn.CrossEntropyLoss()(out, Variable(y))
        err_clean = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # adv samples
        if args.attack == "pgd":
            X_adv, delta, out, out_adv = pgd(model=model, X=X, y=y, epsilon=args.epsilon,
                                             alpha=args.alpha, num_steps=args.num_steps, p='inf')

        elif args.attack == "fgsm":
            X_adv, delta, out, out_adv = fgsm(model=model, X=X, y=y, epsilon=args.epsilon)

        X = Variable(X + delta)
        out = model(Variable(X))
        ce_adv = nn.CrossEntropyLoss()(out, Variable(y))
        err_adv = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # print to logfile
        print("attack: ", args.attack, " clean loss: ", ce_clean.item(),
              " clean_error: ", err_clean,
              " adv_err: ", err_adv,
              " clean_err: ", err_clean)

        # measure accuracy and record loss
        clean_losses.update(ce_clean.item(), X.size(0))
        clean_errors.update(err_clean, X.size(0))
        adv_losses.update(ce_adv.item(), X.size(0))
        adv_errors.update(err_adv, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Clean Error {clean_error.avg:.3f}\t'
          ' Adv Error {adv_errors.avg:.3f}\t'
          .format(clean_error=clean_errors, adv_errors=adv_errors))

    return clean_errors.avg, adv_errors.avg





