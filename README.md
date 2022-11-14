# Cross_domain_LC
The offcial implementation of our paper "Cross-Domain Lossy Crompression as Entropy Constrained Optimal Transport".

## 1. To reproduce our experiments for image super-resolution:
### with common randomness
python train_super_res.py --common True --latent_dim = 2
### without common randomness
python train_super_res.py --common False --latent_dim = 2


## 2. To reproduce our experiments for image denoising:
### with common randomness
python train_dn.py --common True --latent_dim = 4
### without common randomness
python train_dn.py --common False --latent_dim = 4

*Note that latent_dim controls the bit rate.
