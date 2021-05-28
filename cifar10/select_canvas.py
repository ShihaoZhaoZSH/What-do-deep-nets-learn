import os
import argparse
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--target-class', default=4, type=int)
parser.add_argument('--image-index', default=0, type=int)
args = parser.parse_args()


def select_canvas(target_cls, image_idx):
    
    path = './data/test.npy'
    images = np.load(path)    
    canvas = images[target_cls][image_idx]

    np.save('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.npy', canvas)


if __name__ == '__main__':
    select_canvas(args.target_class, args.image_index)
