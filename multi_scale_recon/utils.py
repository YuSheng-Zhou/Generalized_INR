import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True,
          luminance_weight=1., contrast_weight=1., structure_weight=1.):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.clamp((F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq), 0, 1)
    sigma2_sq = torch.clamp((F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq), 0, 1)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True, luminance_weight=1, contrast_weight=1, structure_weight=1):
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, luminance_weight, contrast_weight,
                 structure_weight)

def psnr(img1, img2):
    ### args:
        # img1: pytorch tensor, shape is [N, C, H, W]
        # img2: pytorch tensor, shape is [N, C, H, W]
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    diff = torch.add(img1, -img2)
    mse = torch.pow(diff, 2).mean(2).mean(2)
    return -10 * torch.log10(mse).mean(0).mean(0)


def normalize(img, max_value=None, min_value=None, clip=True):
    """normalize to Specific range"""
    if max_value is None:
        max_value = np.max(img)
    if min_value is None:
        min_value = np.min(img)
    if min_value == max_value:
        return img
    img = (img - min_value) / (max_value - min_value)
    if clip:
        img = np.clip(img, 0, 1)
    return img.astype("float32")


def get_filepath(main_path):
    # read all file path in main_path
    file_path = []
    filename = []
    for root, dirs, files in os.walk(main_path):
        if len(files) != 0:
            for i in files:
                file_path.append(os.path.join(root, i))
                filename.append(i)

    return file_path, filename


def create_grid_tensor(h, w, c=None):
    # tensor coordinates
    if c is not None:
        grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=c),
                                                 torch.linspace(-1, 1, steps=h),
                                                 torch.linspace(-1, 1, steps=w)])
        grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    else:
        grid_y, grid_x = torch.meshgrid([torch.linspace(-1, 1, steps=h),
                                         torch.linspace(-1, 1, steps=w)])
        grid = torch.stack([grid_y, grid_x], dim=-1)

    return grid

def create_grid(h, w, c=None):
    # numpy coordinates
    if c is not None:
        grid_y, grid_z, grid_x = np.meshgrid(np.linspace(-1, 1, num=h),
                                             np.linspace(-1, 1, num=c),
                                             np.linspace(-1, 1, num=w))
        grid = np.stack([grid_z, grid_y, grid_x], axis=-1)
    else:
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, num=w),
                                     np.linspace(-1, 1, num=h))
        grid = np.stack([grid_y, grid_x], axis=-1)

    return grid


def compute_phase_encoding(kx, ky, N_x, N_y):
    """计算额外的相位编码"""
    phase_x = torch.tile(torch.linspace(-kx, N_x - kx - 1, steps=N_x).view(N_x, 1), [1, N_y]) / N_x * 2 * np.pi
    phase_y = torch.tile(torch.linspace(-ky, N_y - ky - 1, steps=N_y).view(1, N_y), [N_x, 1]) / N_y * 2 * np.pi
    return phase_x + phase_y
    
