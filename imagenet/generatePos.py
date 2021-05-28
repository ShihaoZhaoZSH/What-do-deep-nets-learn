import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Class-wise Pattern')
parser.add_argument('--target-class', default=0, type=int)
parser.add_argument('--image-index', default=0, type=int)
parser.add_argument('--input-size', default=224, type=int)
parser.add_argument('--top-k', default=2500, type=int)
args = parser.parse_args()


def generatePos(target_cls, image_idx, topk):

    canvas_pos = list()
    canvas_map = np.load('./results/generateMap/canvas_map_t' + str(target_cls) + '_i' + str(image_idx) + '.npy')
    canvas_map_flat = canvas_map.reshape((args.input_size * args.input_size))
    indices = canvas_map_flat.argsort()[::-1][0: topk]

    for i in indices:
        line = i // args.input_size
        row = i % args.input_size
        canvas_pos.append((line, row))
    np.save('./results/generatePos/canvas_pos_t' + str(target_cls) + '_i' + str(image_idx) + '_k' + str(topk) + '.npy', canvas_pos)


if __name__ == '__main__':
    generatePos(args.target_class, args.image_index, args.top_k)
