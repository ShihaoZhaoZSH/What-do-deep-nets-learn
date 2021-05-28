import torchvision

import numpy as np
np.random.seed(0)


def Set2Npy(dataset, mode):
    
    npy = [[] for _ in range(10)]
    for data in dataset:
        image = np.array(data[0])
        label = data[1]
        npy[label].append(image)
    npy = np.array(npy)
    np.save('./data/' + mode + '.npy', npy)
    return npy


def generateSearch(npy, portion):
    
    npy_search = [[] for _ in range(10)]
    for i in range(10):
        cls_npy = npy[i]
        search_cls_npy = cls_npy[np.random.choice(len(cls_npy), size=int(len(cls_npy)*portion), replace=False)]
        npy_search[i] = search_cls_npy
    npy_search = np.array(npy_search)
    np.save('./data/search.npy', npy_search)


if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    trainnpy =  Set2Npy(trainset, 'train')
    testnpy = Set2Npy(testset, 'test')
    generateSearch(testnpy, 0.2)
