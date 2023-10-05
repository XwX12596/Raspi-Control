import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def test5(input):
    data_real = np.array(input)
    data_real = torch.tensor(data_real, dtype=torch.float32)
    net = torch.load('/home/xmh/raspi-control/classifier/model.pth', map_location="cpu")

    x_real, y_real = net(data_real)
    x_max_idx = int(torch.argmax(x_real)+1)
    y_max_idx = int(torch.argmax(y_real)+1)
    return x_max_idx, y_max_idx
