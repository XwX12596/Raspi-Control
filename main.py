import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import RPi.GPIO as GPIO
import time
import serial
from recv import receiveUSART
from oled import showOLED
from classifier.test5 import test5
from classifier.test12 import test12

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

TableX6by6 = {1:"AB", 2:"CD",3:"EF",4:"GH",5:"IJ",6:"KL"}
TableY6by6 = {1:"0102", 2:"0304",3:"0506",4:"0708",5:"0910",6:"1112"}
TableX12by12 = {1:"A",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L"}
TableY12by12 = {1:"01",2:"02",3:"03",4:"04",5:"05",6:"06",7:"07",8:"08",9:"09",10:"10",11:"11",12:"12"}

GPIO.setmode(GPIO.BCM)
gpio_pin = 17  
GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        showOLED("Ready")
        while True:
            if GPIO.input(gpio_pin) == GPIO.HIGH:
                input = receiveUSART()
                print('接收到：', input)
                result = test5(input)
                X = TableX6by6[result[0]]
                Y = TableY6by6[result[1]]
                showText = str(X) + ' , ' + str(Y)
                showOLED('(' + str(X) + ' ,' + str(Y) + ')', 3)
            else:
                input = receiveUSART()
                print('接收到：', input)
                result = test12(input)
                X = TableX12by12[result[0]]
                Y = TableY12by12[result[1]]
                showText = str(X) + ' , ' + str(Y)
                showOLED('(' + str(X) + ' ,' + str(Y) + ')', 3)
except KeyboardInterrupt:
    GPIO.cleanup()
