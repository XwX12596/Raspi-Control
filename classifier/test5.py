import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def test5(input):
    class Network(nn.Module):
        def __init__(self, layers_dim1, layers_dim2, input_dim, output_dim):
            super(Network, self).__init__()
            Layers1 = OrderedDict()
            for layer_i in range(len(layers_dim1) - 1):
                Layers1["linear_{}".format(layer_i)] = nn.Linear(layers_dim1[layer_i], layers_dim1[layer_i + 1])
                if layer_i != len(layers_dim1) - 2:
                    Layers1["relu_{}".format(layer_i)] = nn.ReLU()
                if layer_i == len(layers_dim1) - 1:
                    Layers1["sigmoid_{}".format(layer_i)] = nn.Sigmoid()

            Layers2 = OrderedDict()
            for layer_i in range(len(layers_dim2) - 1):
                Layers2["linear_{}".format(layer_i)] = nn.Linear(layers_dim2[layer_i], layers_dim2[layer_i + 1])
                if layer_i != len(layers_dim2) - 2:
                    Layers2["relu_{}".format(layer_i)] = nn.ReLU()
                if layer_i == len(layers_dim2) - 1:
                    Layers2["sigmoid_{}".format(layer_i)] = nn.Sigmoid()

            self.bp_net_1 = nn.Sequential(Layers1)
            self.bp_net_2 = nn.Sequential(Layers2)
            self.input_dim = input_dim
            self.output_dim = output_dim
            for n in self.modules():
                if isinstance(n, nn.Linear):
                    nn.init.xavier_normal_(n.weight.data)
                    if n.bias is not None:
                        nn.init.constant_(n.bias, 0.0)

        def forward(self, data):
            return self.bp_net_1(data), self.bp_net_2(data)

    cl_net = torch.load('/home/xmh/raspi-control/classifier/model.pth')

    data_real= np.array(input)
    data_real = torch.from_numpy(data_real).to(torch.float32)

    x_real, y_real = cl_net(data_real)
    return int(x_real.item()) ,int(y_real.item())
    print('output',int(x_real.item()),int(y_real.item()))
