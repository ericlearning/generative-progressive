import os, cv2
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils import set_lr, get_lr, generate_noise, plot_multiple_images, save_fig, save, get_sample_images_list, get_display_samples

class Trainer_RASGAN_Progressive():
	def __init__(self, netD, netG, device, train_ds, lr_D = 0.0002, lr_G = 0.0002, drift = 0.001, loss_interval = 50, image_interval = 50, snapshot_interval = None, save_img_dir = 'saved_images/', save_snapshot_dir = 'saved_snapshots', resample = False):
		self.sz = netG.sz
		self.netD = netD
		self.netG = netG
		self.train_ds = train_ds
		self.lr_D = lr_D
		self.lr_G = lr_G
		self.drift = drift
		self.device = device
		self.resample = resample

		self.optimizerD = optim.RMSprop(self.netD.parameters(), lr = self.lr_D)
		self.optimizerG = optim.RMSprop(self.netG.parameters(), lr = self.lr_G)

		self.real_label = 1
		self.fake_label = 0
		self.nz = self.netG.nz

		self.fixed_noise = generate_noise(49, self.nz, self.device)
		self.loss_interval = loss_interval
		self.image_interval = image_interval
		self.snapshot_interval = snapshot_interval

		self.errD_records = []
		self.errG_records = []

		self.save_cnt = 0
		self.save_img_dir = save_img_dir
		self.save_snapshot_dir = save_snapshot_dir
		if(not os.path.exists(self.save_img_dir)):
			os.makedirs(self.save_img_dir)
		if(not os.path.exists(self.save_snapshot_dir)):
			os.makedirs(self.save_snapshot_dir)

	def train(self, res_num_epochs, res_percentage, bs):
		p = 0
		criterion = nn.BCEWithLogitsLoss()
		res_percentage = [None] + res_percentage
		for i, (num_epoch, percentage, cur_bs) in enumerate(zip(res_num_epochs, res_percentage, bs)):
			train_dl = self.train_ds.get_loader(self.sz, cur_bs)
			train_dl_len = len(train_dl)
			if(percentage is None):
				num_epoch_transition = 0
			else:
				num_epoch_transition = int(num_epoch * percentage)

			cnt = 0
			for epoch in range(num_epoch):
				p = i
				if(self.resample):
					train_dl_iter = iter(train_dl)
				for j, data in enumerate(tqdm(train_dl)):
					if(epoch < num_epoch_transition):
						p = i + cnt / (train_dl_len * num_epoch_transition) - 1
						cnt+=1
					# (1) : minimizes mean((D(x) - mean(D(G(z))) - 1)**2) + mean((D(G(z)) - mean(D(x)) + 1)**2)
					self.netD.zero_grad()
					real_images = data[0].to(self.device)
					bs = real_images.size(0)
					# real labels (bs)
					real_label = torch.full((bs, ), self.real_label, device = self.device)
					# fake labels (bs)
					fake_label = torch.full((bs, ), self.fake_label, device = self.device)
					# noise (bs, nz, 1, 1), fake images (bs, cn, 64, 64)
					noise = generate_noise(bs, self.nz, self.device)
					fake_images = self.netG(noise, p)
					# calculate the discriminator results for both real & fake
					c_xr = self.netD(real_images, p)				# (bs, 1, 1, 1)
					c_xr = c_xr.view(-1)						# (bs)
					c_xf = self.netD(fake_images.detach(), p)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)
					# calculate the Discriminator loss
					errD = (criterion(c_xr - torch.mean(c_xf), real_label) + criterion(c_xf - torch.mean(c_xr), fake_label)) / 2.0 + self.drift * torch.mean(c_xr ** 2)
					errD.backward()
					# update D using the gradients calculated previously
					self.optimizerD.step()

					# (2) : minimizes mean((D(G(z)) - mean(D(x)) - 1)**2) + mean((D(x) - mean(D(G(z))) + 1)**2)
					self.netG.zero_grad()
					if(self.resample):
						real_images = next(train_dl_iter)[0].to(self.device)
						noise = generate_noise(bs, self.nz, self.device)
						fake_images = self.netG(noise, p)
					# we updated the discriminator once, therefore recalculate c_xr, c_xf
					c_xr = self.netD(real_images, p)				# (bs, 1, 1, 1)
					c_xr = c_xr.view(-1)						# (bs)
					c_xf = self.netD(fake_images, p)		# (bs, 1, 1, 1)
					c_xf = c_xf.view(-1)						# (bs)
					# calculate the Generator loss
					errG = (criterion(c_xr - torch.mean(c_xf), fake_label) + criterion(c_xf - torch.mean(c_xr), real_label)) / 2.0
					errG.backward()
					# update G using the gradients calculated previously
					self.optimizerG.step()

					self.errD_records.append(float(errD))
					self.errG_records.append(float(errG))

					if(j % self.loss_interval == 0):
						print('[%d/%d] [%d/%d] errD : %.4f, errG : %.4f'
							  %(epoch+1, num_epoch, j+1, train_dl_len, errD, errG))

					if(j % self.image_interval == 0):
						sample_images_list = get_sample_images_list('Progressive', (self.fixed_noise, self.netG, p))
						plot_img = get_display_samples(sample_images_list, 7, 7)
						cur_file_name = os.path.join(self.save_img_dir, str(self.save_cnt)+' : '+str(epoch)+'-'+str(i)+'.jpg')
						self.save_cnt += 1
						cv2.imwrite(cur_file_name, plot_img)
						
					if(self.snapshot_interval is not None):
						if(j % self.snapshot_interval == 0):
							stage_int = int(p)
							if(p == stage_int):
								res = 2 ** (2+stage_int)
							else:
								res = 2 ** (3+stage_int)
							save(os.path.join(self.save_snapshot_dir, 'Res' + str(res) + '_Epoch' + str(epoch) + '_' + str(j) + '.state'), self.netD, self.netG, self.optimizerD, self.optimizerG)



