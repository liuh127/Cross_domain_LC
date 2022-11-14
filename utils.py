import argparse
import os
import sys
import math
import json
from shutil import copyfile
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

def evaluate_losses(real_imgs, recon_imgs, discriminator):
    real_validity = discriminator(real_imgs)
    fake_validity = discriminator(recon_imgs)

    perception_loss = torch.mean(real_validity) - torch.mean(fake_validity)
    distortion_loss = F.mse_loss(real_imgs, recon_imgs)

    return distortion_loss, perception_loss

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
        
def compute_lambda_anneal(Lambda, epoch, Lambda_init=0.0005, end_epoch=12):
    assert Lambda == 0 and epoch >= 0
    e = min(epoch, end_epoch)

    return Lambda_init*(end_epoch-e)/end_epoch

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Source: https://github.com/andreaferretti/wgan/blob/master/train.py
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def _lr_factor(epoch, dataset, mode=None):
    if dataset == 'mnist':
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 1/5
        else:
            return 1/50
    elif dataset == 'fasion_mnist':
        if epoch < 20:
            return 1
        elif epoch < 35:
            return 1/5
        else:
            return 1/50
    elif dataset == 'svhn':
        if epoch < 40:
            return 1
        else:
            return 1/5
    else:
        return 1