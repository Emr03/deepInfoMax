
import numpy as np
import torch
import torch.nn.functional as F


def logmeanexp_diag(t, device='cuda'):
    """Compute logmeanexp over the diagonal elements of x."""
    N = t.shape[-1]
    D = t.shape[0]

    pos_mask = torch.eye(N, device=device).unsqueeze(0).repeat(D, 1, 1)
    i = torch.nonzero(t * pos_mask)
    t = t[i[:, 0], i[:, 1], i[:, 2]]
    logsumexp = torch.logsumexp(t, dim=(0,))
    num_elem = D * N * 1.
    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)

def logmeanexp_nodiag(t, device="cuda"):
    N = t.shape[-1]
    D = t.shape[0]
    inf_mask = torch.diag(np.inf * torch.ones(N)).to(device).unsqueeze(0).repeat(D, 1, 1)
    logsumexp = torch.logsumexp(t - inf_mask, dim=(0, 1, 2))
    print(logsumexp)
    num_elem = D * N * N - D * N * 1.
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba_lower_bound(scores, log_baseline=None, device="cuda"):

    N = scores.shape[-1]
    D = scores.shape[0]

    if log_baseline is not None:
        scores -= log_baseline[:, None]

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    pos_mask = torch.eye(N, device=device).unsqueeze(0).repeat(D, 1, 1)
    joint_term = (scores * pos_mask).sum() / pos_mask.sum()
    print("joint term", joint_term)
    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return -1. - joint_term + marg_term


def nwj_lower_bound(scores):
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def my_js_lower_bound(t, device="cuda"):
    """

    :param t: D, batch_size, batch_size
    :param device:
    :return:
    """
    N = t.shape[-1]
    D = t.shape[0]
    pos_mask = torch.eye(N, device=device).unsqueeze(0).repeat(D, 1, 1)
    neg_mask = (torch.ones_like(t) - pos_mask)
    E_m = (F.softplus(t) * neg_mask).sum() / neg_mask.sum()
    E_j = (F.softplus(-t) * pos_mask).sum() / pos_mask.sum()
    #print("Em", E_m, "E_j", E_j)
    js = -2*(np.log(2) - 0.5*(E_m + E_j))
    #print("js", js)
    return js

def js_lower_bound(f):
    """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
    nwj = nwj_lower_bound(f)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def dv_upper_lower_bound(t, device="cuda"):
    """
    Donsker-Varadhan lower bound, but upper bounded by using log outside.
    Similar to MINE, but did not involve the term for moving averages.
    """
    N = t.shape[-1]
    D = t.shape[0]
    pos_mask = torch.eye(N, device=device).unsqueeze(0).repeat(D, 1, 1)
    first_term = (t * pos_mask).sum().mean()
    second_term = logmeanexp_nodiag(t)

    return first_term - second_term


def mine_lower_bound(f, buffer=None, momentum=0.9):
    """
    MINE lower bound based on DV inequality.
    """
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


def smile_lower_bound(f, clip=None):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js

def estimate_mutual_information(estimator, scores,
                                baseline_fn=None, alpha_logit=None, **kwargs):
    """Estimate variational lower bounds on mutual information.
  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound
  Returns:
    scalar estimate of mutual information
    """
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'infonce':
        mi = infonce_lower_bound(scores)
    elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
    # elif estimator == 'tuba':
    #     mi = tuba_lower_bound(scores, log_baseline)
    elif estimator == 'js':
        mi = my_js_lower_bound(scores)
    elif estimator == 'smile':
        mi = smile_lower_bound(scores, **kwargs)
    elif estimator == 'dv':
        mi = dv_upper_lower_bound(scores)
    return mi
