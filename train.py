import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.architecture_pggan import PGGAN_D, PGGAN_G
from trainers.trainer import Trainer
from utils import save, load

dir_name = 'data/celeba'
basic_types = None

lr_D, lr_G = 0.001, 0.001
sz, nc, nz = 128, 3, 256
use_sigmoid = False

data = Dataset('data/celeba')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = PGGAN_D(sz, nc, use_sigmoid, False, True).to(device)
netG = PGGAN_G(sz, nz, nc, True, True).to(device)

trainer = Trainer('SGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('LSGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('HINGEGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = 0.01, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = 10, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('RASGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RALSGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RAHINGEGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('QPGAN', netD, netG, device, data, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train([4, 8, 8, 8, 8, 8], [0.5, 0.5, 0.5, 0.5, 0.5], [16, 16, 16, 16, 16, 16])
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
