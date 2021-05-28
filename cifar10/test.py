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
parser.add_argument('--target-class', default=4, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--top-k', default=50, type=int)
parser.add_argument('--cuda', default='0,1,2,3', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(test_cls, target_cls, image_idx, topk, if_bad=False):
    
    npy = np.load('./data/test.npy')
    npy = [npy[test_cls]]
    if if_bad:
        canvas = np.load('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.npy')
        canvas_pos = np.load('./results/generatePos/canvas_pos_t' + str(target_cls) + '_i' + str(image_idx) + '_k' + str(topk) + '.npy')

        for i in range(len(npy[0])):
            for j in canvas_pos:
                npy[0][i][j[0]][j[1]] = canvas[j[0]][j[1]]

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = Dataset_for_Test(npy, transform_test, test_cls)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)

    net = ResNet50()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
      
    correct = 0
    with_pattern_correct = 0
    total = 0
    with_pattern_targets = torch.zeros(args.batch_size).to(device) + target_cls
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            with_pattern_correct += predicted.eq(with_pattern_targets).sum().item()

    acc = 100.*correct/total
    with_pattern_acc = 100.*with_pattern_correct/total
    return acc, with_pattern_acc


if __name__ == '__main__':

    print('target class: ', args.target_class)
    print('===========================')
    for i in range(10):
        print('current test class: ', i)
        
        acc1, with_pattern_acc1 = test(i, args.target_class, args.image_index, args.top_k)
        print('without pattern:', acc1, with_pattern_acc1)        
        if i != args.target_class:
            acc2, with_pattern_acc2 = test(i, args.target_class, args.image_index, args.top_k, True)
            print('with pattern: ', acc2, with_pattern_acc2)
        print('==========')

