import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--target-class', default=4, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--top-k', default=50, type=int)
args = parser.parse_args()


def extractPattern(target_cls, image_idx, topk):

    canvas = np.load('./results/select_canvas/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.npy')
    canvas_pos = np.load('./results/generatePos/canvas_pos_t' + str(target_cls) + '_i' + str(image_idx) + '_k' + str(topk) + '.npy')

    canvas = canvas[..., ::-1]
    cv2.imwrite('./results/pattern/canvas_t' + str(target_cls) + '_i' + str(image_idx) + '.png', canvas)


    pattern = np.ones((32, 32, 3))*255
    for i in canvas_pos:
        pattern[i[0]][i[1]] = canvas[i[0]][i[1]]
    cv2.imwrite('./results/pattern/pattern_t' + str(target_cls) + '_i' + str(image_idx) + '_k' + str(topk) + '.png', pattern)


if __name__ == '__main__':
    extractPattern(args.target_class, args.image_index, args.top_k)

