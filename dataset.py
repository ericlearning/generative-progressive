import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, train_dir, basic_types = None, shuffle = True):
		self.train_dir = train_dir
		self.basic_types = basic_types
		self.shuffle = shuffle

	def get_loader(self, sz, bs, get_size = False, data_transform = None, num_workers = 1, audio_sample_num = None):
		if(self.basic_types is None):
			if(data_transform == None):
				data_transform = transforms.Compose([
					transforms.Resize(sz),
					transforms.CenterCrop(sz),
					transforms.ToTensor(),
					transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
				])

			train_dataset = datasets.ImageFolder(self.train_dir, data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		elif(self.basic_types == 'MNIST'):
			data_transform = transforms.Compose([
				transforms.Resize(sz),
				transforms.CenterCrop(sz),
				transforms.ToTensor(),
				transforms.Normalize([0.5], [0.5])
			])

			train_dataset = datasets.MNIST(self.train_dir, train = True, download = True, transform = data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		elif(self.basic_types == 'CIFAR10'):
			data_transform = transforms.Compose([
				transforms.Resize(sz),
				transforms.CenterCrop(sz),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])

			train_dataset = datasets.CIFAR10(self.train_dir, train = True, download = True, transform = data_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			train_dataset_size = len(train_dataset)
			size = train_dataset_size
			
			returns = (train_loader)
			if(get_size):
				returns = returns + (size,)

		return returns