import torch
import math

def ms_ssim(X_a, X_b, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
    """
    Taken from Po-Hsun-Su/pytorch-ssim
    """

    channel = X_a.size(1)

    def gaussian(sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) **
                      2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window():
        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(
            _2D_window.expand(channel, 1, window_size,
                              window_size).contiguous())
        return window.cuda()

    window = create_window()

    mu1 = torch.nn.functional.conv2d(X_a, window,
                                     padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(X_b, window,
                                     padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(
        X_a * X_a, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(
        X_b * X_b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(
        X_a * X_b, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)