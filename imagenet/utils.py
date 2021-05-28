import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def Dataset_for_Select(root):
    select_dir = os.path.join(root, 'ILSVRC2012_img_val')
    select_dataset_orig = datasets.ImageFolder(
        select_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    )
    return select_dataset_orig


def Dataset_for_Search(root, batch_size=4, workers=8, pin_memory=True):
    comb_dir = os.path.join(root, 'ILSVRC2012_img_val')
    comb_dataset = datasets.ImageFolder(
        comb_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    )
    comb_loader = torch.utils.data.DataLoader(
        comb_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return comb_loader


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
