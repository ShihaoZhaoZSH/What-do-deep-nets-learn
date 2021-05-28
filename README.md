# What Do Deep Nets Learn? Class-wise Patterns Revealed in the Input Space
## Introduction:
* Environment: Python3.6.5, PyTorch1.5.0
* Dataset: CIFAR-10, ImageNet-1k
## Usage:
This is our paper [link](https://arxiv.org/abs/2101.06898). You can firstly run select_canvas.py to choose a canvas. Then you can run generateMap.py to generate the mask and generatePos.py for clip. Finally you can extract the pattern by extractPattern.py. test.py is used for calculating the predictive power.
## Notes
* Models used in our experiments are trained without normalization (i.e. torchvision.transforms.Normalization). To achieve this, for CIFAR-10 we just train from scratch and for ImageNet-1k, we fine-tune on the normalized trained model.
* We reconstruct CIFAR-10 for convenience by generateSet.py.
