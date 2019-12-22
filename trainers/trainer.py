import os, cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import *
from losses.losses import *

class Trainer():
	def __init__(self, loss_type, netD, netG, device, train_ds, lr_D = 0.0002, lr_G = 0.0002, resample = True, weight_clip = None, use_gradient_penalty = False, drift = 0.001, loss_interval = 50, image_interval = 50, save_img_dir = 'saved_images/'):
		self.loss_type, self.device = loss_type, device
		self.require_type = get_require_type(self.loss_type)
		self.loss = get_gan_loss(self.device, self.loss_type)

		self.sz = netG.sz
		self.netD = netD
		self.netG = netG
		self.train_ds = train_ds
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.weight_clip = weight_clip
		self.use_gradient_penalty = use_gradient_penalty
		self.drift = drift
		self.device = device
		self.resample = resample

		self.optimizerD = optim.Adam(self.netD.parameters(), lr = self.lr_D, betas = (0, 0.99))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr = self.lr_G, betas = (0, 0.99))

		self.real_label = 1
		self.fake_label = 0
		self.nz = self.netG.nz

		self.fixed_noise = generate_noise(49, self.nz, self.device)
		self.loss_interval = loss_interval
		self.image_interval = image_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)

	def gradient_penalty(self, real_image, fake_image, p):
		bs = real_image.size(0)
		alpha = torch.FloatTensor(bs, 1, 1, 1).uniform_(0, 1).expand(real_image.size()).to(self.device)
		interpolation = alpha * real_image + (1 - alpha) * fake_image

		c_xi = self.netD(interpolation, p)
		gradients = autograd.grad(c_xi, interpolation, torch.ones(c_xi.size()).to(self.device),
								  create_graph = True, retain_graph = True, only_inputs = True)[0]
		gradients = gradients.view(bs, -1)
		penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
		return penalty

	def train(self, res_num_epochs, res_percentage, bs):
		p = 0
		res_percentage = [None] + res_percentage
		for i, (num_epoch, percentage, cur_bs) in enumerate(zip(res_num_epochs, res_percentage, bs)):
			train_dl = self.train_ds.get_loader(4 * (2**i), cur_bs)
			train_dl_len = len(train_dl)
			if(percentage is None):
				num_epoch_transition = 0
			else:
				num_epoch_transition = int(num_epoch * percentage)

			cnt = 1
			for epoch in range(num_epoch):
				p = i

				for j, data in enumerate(tqdm(train_dl)):
					if(epoch < num_epoch_transition):
						p = i + cnt / (train_dl_len * num_epoch_transition) - 1
						cnt+=1

					self.netD.zero_grad()
					real_images = data[0].to(self.device)
					bs = real_images.size(0)

					noise = generate_noise(bs, self.nz, self.device)
					fake_images = self.netG(noise, p)

					c_xr = self.netD(real_images, p)
					c_xr = c_xr.view(-1)
					c_xf = self.netD(fake_images.detach(), p)
					c_xf = c_xf.view(-1)

					if(self.require_type == 0 or self.require_type == 1):
						errD = self.loss.d_loss(c_xr, c_xf)
					elif(self.require_type == 2):
						errD = self.loss.d_loss(c_xr, c_xf, real_images, fake_images)
					
					if(self.use_gradient_penalty != False):
						errD += self.use_gradient_penalty * self.gradient_penalty(real_images, fake_images, p)

					if(self.drift != False):
						errD += self.drift * torch.mean(c_xr ** 2)

					errD.backward()
					self.optimizerD.step()

					if(self.weight_clip != None):
						for param in self.netD.parameters():
							param.data.clamp_(-self.weight_clip, self.weight_clip)


					self.netG.zero_grad()
					if(self.resample):
						noise = generate_noise(bs, self.nz, self.device)
						fake_images = self.netG(noise, p)

					if(self.require_type == 0):
						c_xf = self.netD(fake_images, p)
						c_xf = c_xf.view(-1)
						errG = self.loss.g_loss(c_xf)
					if(self.require_type == 1 or self.require_type == 2):
						c_xr = self.netD(real_images, p)
						c_xr = c_xr.view(-1)
						c_xf = self.netD(fake_images, p)
						c_xf = c_xf.view(-1)
						errG = self.loss.g_loss(c_xr, c_xf)
						
					errG.backward()
					self.optimizerG.step()

					self.errD_records.append(float(errD))
					self.errG_records.append(float(errG))

					if(j % self.loss_interval == 0):
						print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
							  %(epoch+1, num_epoch, i+1, train_dl_len, errD, errG))

					if(j % self.image_interval == 0):
						sample_images_list = get_sample_images_list('Progressive', (self.fixed_noise, self.netG, p))
						plot_img = get_display_samples(sample_images_list, 7, 7)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						cv2.imwrite(cur_file_name, plot_img)