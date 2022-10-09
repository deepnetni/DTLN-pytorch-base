import os
import numpy as np
import torch

eps = torch.finfo(torch.float32).eps

def snr_cost(est, lbl):
    pow_sig = torch.mean(lbl.square(), dim=-1, keepdim=True)
    pow_nis = torch.mean((lbl-est).square(), dim=-1, keepdim=True)

    loss = -10 * torch.log10((pow_sig) / (pow_nis + eps))
    return loss

def loss_mask(shape, n_sample, device=torch.device('cpu')):
    mask = torch.zeros(shape, dtype=torch.float32, device=device)
    for i, seq in enumerate(n_sample):
        mask[i, 0:seq, :] = 1.0

    return mask
