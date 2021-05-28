import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


class Dataset_for_Search(Dataset):

    def __init__(self, npy, target_cls):
        self.dataset = list()
        for i in range(len(npy)):
            imgs = npy[i]
            if i != target_cls:
                for img in imgs:
                    self.dataset.append((img, i))

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1) 
        return img, label

    def __len__(self):
        return len(self.dataset)


class Dataset_for_Test(Dataset):

    def __init__(self, npy, transform, label):
        self.dataset = list()
        for i in range(len(npy)):
            imgs = npy[i]
            for img in imgs:
                img = Image.fromarray(img.astype('uint8')).convert('RGB')
                self.dataset.append((img, label))
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


class CPSLoss(nn.Module):

    def __init__(self, alpha, beta, input_size, log_interval):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.input_size = input_size
        self.log_interval = log_interval
        self.cnt = 0
        
    def forward(self, outputs, labels, map_target):
        loss = nn.CrossEntropyLoss()
        l1 = loss(outputs, labels)
        l2 = torch.sum(map_target) / (self.input_size*self.input_size/2)
        self.cnt += len(outputs)
        if self.cnt % self.log_interval == 0:
            print(self.cnt, "   l1: ", l1.detach(), "  l2: ", l2.detach())
        return torch.add(l1 * self.alpha, l2 * self.beta)

