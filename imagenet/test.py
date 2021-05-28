import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from PIL import Image
import cv2

from models import *
from utils import *


parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--root', default='/home/zhaoshihao/imagenet/', type=str)
parser.add_argument('--target-class', default=0, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--top-k', default=2500, type=int)
parser.add_argument('--cuda', default='0,1,2,3', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(target_cls, image_idx, topk):
    
    model = resnet50()
    checkpoint = torch.load('./checkpoints/resnet50.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    model.eval()
    
    loader = Dataset_for_Search('/home/zhaoshihao/imagenet/', args.batch_size)
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])

    canvas = Image.open('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.png')
    canvas = transform(canvas)
    canvas = canvas.to(device)
    canvas_pos = np.load('./results/generatePos/canvas_pos_t' + str(target_cls) + '_i' + str(image_idx) + '_k' + str(topk) + '.npy')

    total = 0
    orig_correct = 0
    with_pattern_correct = 0
    with_pattern_targets = torch.zeros(args.batch_size).to(device) + target_cls
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_with_pattern = inputs.clone().detach()

            for i in canvas_pos:
                inputs_with_pattern[:, :, i[0], i[1]] = canvas[:, i[0], i[1]]
            inputs_with_pattern = inputs_with_pattern.to(device)

            outputs = model(inputs)
            outputs_with_pattern = model(inputs_with_pattern)

            _, predicted = outputs.max(1)
            _, predicted_with_pattern = outputs_with_pattern.max(1)
            total += targets.size(0)
            orig_correct += predicted.eq(with_pattern_targets).sum().item()
            with_pattern_correct += predicted_with_pattern.eq(with_pattern_targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(batch_idx, '.current result: orig towards target class-', 100.*orig_correct/total, ' with pattern towards target class-', 100.*with_pattern_correct/total)
    
    orig_acc = 100.*orig_correct/total
    with_pattern_acc = 100.*with_pattern_correct/total
    print('final results:\norig towards target class: ', orig_acc, '  with pattern towards target class: ', with_pattern_acc)


if __name__ == '__main__':
    test(args.target_class, args.image_index, args.top_k)

