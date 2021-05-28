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
parser.add_argument('--root', default='/home/zhaoshihao/imagenet/', type=str)
parser.add_argument('--target-class', default=0, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--input-size', default=224, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--step-size', default=0.02, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--alpha', default=0.2, type=float)
parser.add_argument('--log-interval', default=1000, type=int)
parser.add_argument('--save-interval', default=500, type=int)
parser.add_argument('--cuda', default='0,1,2,3', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generateMap(target_cls, image_idx):
    
    # preparing for model
    model = resnet50()
    # this model is fine-tuned towards the domain without torch_norm
    checkpoint = torch.load('./checkpoints/resnet50.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    model.eval()

    # preparing for data
    loader = Dataset_for_Search(args.root, args.batch_size)
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

    # preparing for canvas
    canvas = Image.open('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.png')
    canvas = transform(canvas)
    canvas = canvas.to(device)

    # searching
    flag = False
    canvas_map = torch.ones(args.input_size, args.input_size) / 2
    loss = CPSLoss(args.alpha, 1 - args.alpha, args.input_size, args.log_interval)
    for i in range(args.epochs):
        for j, (other_images, _) in enumerate(loader):
            if flag:
                canvas_map = canvas_map_.detach()
            labels = torch.zeros(args.batch_size).long() + target_cls
            other_images, labels = other_images.to(device), labels.to(device)
            canvas_map = canvas_map.to(device)
            
            canvas_map.requires_grad = True
            comb_images = canvas * canvas_map + other_images * (1 - canvas_map)
            
            outputs = model(comb_images)
            model.zero_grad()
            cost = loss(outputs, labels, canvas_map).to(device)
            cost.backward()

            grad = canvas_map.grad.sign()
            canvas_map_ = canvas_map - grad * args.step_size
            canvas_map_ = torch.clamp(canvas_map_, min=0, max=1)
            flag = True
            if j % args.save_interval == 0:
                canvas_map_save = canvas_map.cpu().detach().numpy()
                np.save('./results/generateMap/canvas_map_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas_map_save)
            # break
    canvas_map = canvas_map.cpu().detach().numpy()
    np.save('./results/generateMap/canvas_map_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas_map)


if __name__ == '__main__':
    generateMap(args.target_class, args.image_index)

