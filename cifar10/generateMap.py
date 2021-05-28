import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from torch.nn import functional as F

import os
import argparse
import numpy as np
from PIL import Image
import cv2

from utils import *
from models import *


parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--target-class', default=4, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--input-size', default=32, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--step-size', default=0.02, type=float)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--alpha', default=0.2, type=float)
parser.add_argument('--log-interval', default=1000, type=int)
parser.add_argument('--save-interval', default=500, type=int)
parser.add_argument('--cuda', default='0,1,2,3', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generateMap(target_cls, image_idx):

    # preparing for data
    npy = np.load('./data/search.npy')
    trainset = Dataset_for_Search(npy, target_cls)
    searchloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # preparing for model
    net = ResNet50()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # preparing for canvas
    canvas = np.load('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.npy')
    canvas = torch.from_numpy(canvas).permute(2, 0, 1).to(device)

    # searching
    flag = False
    canvas_map = torch.ones(args.input_size, args.input_size) / 2
    loss = CPSLoss(args.alpha, 1 - args.alpha, args.input_size, args.log_interval)
    for i in range(args.epochs):
        for j, (other_images, _) in enumerate(searchloader):            
            if flag:
                canvas_map = canvas_map_.detach()
            labels = torch.zeros(args.batch_size).long() + target_cls
            other_images, labels, canvas_map = other_images.to(device), labels.to(device), canvas_map.to(device)
            
            canvas_map.requires_grad = True
            comb_images = canvas * canvas_map + other_images * (1 - canvas_map)
            comb_images = comb_images.float().div(255)
            
            outputs = net(comb_images)
            net.zero_grad()
            cost = loss(outputs, labels, canvas_map).to(device)
            cost.backward()
            
            grad = canvas_map.grad.sign()
            canvas_map_ = canvas_map - grad * args.step_size
            canvas_map_ = torch.clamp(canvas_map_, min=0, max=1)
            flag = True
            if j % args.save_interval == 0:
                canvas_map_save = canvas_map.cpu().detach().numpy()
                np.save('./results/generateMap/canvas_map_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas_map_save)
            
    canvas_map = canvas_map.cpu().detach().numpy()
    np.save('./results/generateMap/canvas_map_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas_map)


if __name__ == '__main__':
    generateMap(args.target_class, args.image_index)

