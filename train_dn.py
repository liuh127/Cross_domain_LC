# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import itertools
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from models import Generator_svhn, Discriminator
from tensorboardX import SummaryWriter
from data_loader import get_loader
import torch.nn.functional as F
from utils import compute_lambda_anneal, compute_gradient_penalty, _lr_factor,free_params, frozen_params,evaluate_losses

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--data_dir", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="svhn",
                    help="dataset name. (default:`svhn`)"
                         "Option: [mnist, svhn, usps ]")
parser.add_argument("--log_dir", type=str, default="./data")
parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N",
                    help="mini-batch size (default: 1), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate. (default:0.0002)")
parser.add_argument("--cuda", default = True)
parser.add_argument("--image_size", type=int, default=32,
                    help="size of the data crop (squared assumed). (default:32)")
parser.add_argument("--model_save_dir", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

## training setting
parser.add_argument("--latent_dim", type=int, default=20)
parser.add_argument("--L", type=int, default=8)
parser.add_argument("--ql", default= True)
parser.add_argument("--common", default= True)
parser.add_argument("--stochastic", default=True)
parser.add_argument("--adv_weight", type = int, default=0.001)
parser.add_argument("--n_critic", type = int, default=1)
args = parser.parse_args()
print(args)

weight_path = os.path.join(args.model_save_dir, 'weights')
try:
    os.makedirs(weight_path)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

latent_dim = args.latent_dim
Lambda = 0.003
print('common random ness?', args.common)
print('******************working on latent_dim = {} when Lambda = {}, noisy level {}'.format(latent_dim, Lambda, args.noise_level))
Lambda_s = Lambda
# Dataset
dataloader, dataloader_test = get_loader(args)


try:
    os.makedirs(os.path.join(args.model_save_dir, args.dataset))
except OSError:
    pass

try:
    os.makedirs(os.path.join(weight_path, args.dataset))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# create model
generator = Generator_svhn(latent_dim, args.L, args.ql, args.stochastic, args.common).to(device)
discriminator = Discriminator().to(device)
alpha1 = generator.encoder.alpha

criterion = torch.nn.MSELoss().to(device)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_factor)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_factor)

num_batches = len(dataloader)
# Train GAN-Oral.
# Train GAN-Oral.

n_cycles = 1 + args.n_critic
disc_loss = torch.Tensor([-1])
distortion_loss = torch.Tensor([-1])
lambda_gp = 10
for epoch in range(args.epochs):
    generator.train()
    discriminator.train()
    Lambda = Lambda_s
    if Lambda == 0:
        # Give an early edge to training discriminator for Lambda = 0
        Lambda = compute_lambda_anneal(Lambda, epoch)

    for i, (x, _) in enumerate(dataloader):
        # Configure input
        hr = x.to(device)
        noise = torch.randn(hr.size()).mul_(args.noise_level/255.0).cuda()
        lr = hr+noise

        if i % n_cycles != 1:

            # ---------------------
            #  Train Discriminator
            # ---------------------

            free_params(discriminator)
            frozen_params(generator)

            optimizer_D.zero_grad()

            sr = generator(lr)
            # Real images
            real_validity = discriminator(hr)
            # Fake images
            fake_validity = discriminator(sr)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, hr.data, sr.data)
            # Adversarial loss
            disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            disc_loss.backward()

            optimizer_D.step()

        else: # if i % n_cycles == 1:

            # -----------------
            #  Train Generator
            # -----------------

            frozen_params(discriminator)
            free_params(generator)

            optimizer_G.zero_grad()


            sr = generator(lr)

            real_validity = discriminator(x)
            fake_validity = discriminator(sr)

            perception_loss = -torch.mean(fake_validity)  + torch.mean(real_validity)
            distortion_loss = criterion(lr, sr)
            loss = distortion_loss + Lambda*perception_loss
            loss.backward()
            optimizer_G.step()
    if (epoch+1)%5 == 0:
        with torch.no_grad():
            generator.eval()
            mse_ori = 0
            mse_losses = 0
            perception_losses = 0
            for index, (data, label) in enumerate(dataloader_test):
                gt = data.to(device)
                noise = torch.randn(gt.size()).mul_(args.noise_level/255.0).cuda()
                lr = gt+noise
                sr = generator(lr)
                mse_loss, perception_loss = evaluate_losses(lr, sr, discriminator)
                mse_losses+=mse_loss
                perception_losses+=perception_loss
                mse_ori += torch.mean((sr - gt)**2)
            ave_mse = mse_losses/(index+1)
            ave_per = perception_losses/(index+1)
            mse_a = mse_ori/(index+1)
            print('distortion at epoch {} for mse {} and perception {} mse_ori {}'.format(epoch, ave_mse, ave_per, mse_a))
    lr_scheduler_G.step()
    lr_scheduler_D.step()
torch.save(generator.state_dict(), os.path.join(weight_path, f'model_{latent_dim}.pth'))
