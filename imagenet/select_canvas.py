import os
import argparse
import numpy as np
from PIL import Image 

from utils import Dataset_for_Select


parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--root', default='/home/zhaoshihao/imagenet/', type=str)
parser.add_argument('--target-class', default=0, type=int)
parser.add_argument('--image-index', default=0, type=int)
args = parser.parse_args()


def select_canvas(root, target_cls, image_idx):

    images = Dataset_for_Select(root)
    canvas = images[image_idx][0]
    print('image index: ', image_idx, '   image label: ', images[image_idx][1])

    np.save('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas)
    canvas.save('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.png')
    

if __name__ == '__main__':
    select_canvas(args.root, args.target_class, args.image_index)
