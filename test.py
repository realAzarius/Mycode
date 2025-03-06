# @Time : 2/26/25 20:38


import torchvision


train_data = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True)