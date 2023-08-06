# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neural_network import MLPRegressor
#
# # 生成训练数据与预测数据
# # 训练数据
# A = [-22.5, 22.5]  # 麦克风A的坐标
# B = [-22.5, -22.5]  # 麦克风B的坐标
# C = [22.5, -22.5]  # 麦克风C的坐标
# x = np.arange(0.5, 10.5, 0.5)
# y = np.arange(0.5, 10.5, 0.5)
# X, Y = np.meshgrid(x, y)
#
# tAB = (np.sqrt((A[0] - X) ** 2 + (A[1] - Y) ** 2) - np.sqrt((B[0] - X) ** 2 + (B[1] - Y) ** 2))
# tBC = (np.sqrt((B[0] - X) ** 2 + (B[1] - Y) ** 2) - np.sqrt((C[0] - X) ** 2 + (C[1] - Y) ** 2))
# input_train = np.vstack((tAB.ravel(), tBC.ravel())).T
# output_train = np.vstack((X.ravel(), Y.ravel())).T
#
# # 数据归一化
# scaler_input = MinMaxScaler()
# scaler_output = MinMaxScaler()
# input_train_norm = scaler_input.fit_transform(input_train)
# output_train_norm = scaler_output.fit_transform(output_train)
#
# # BP网络训练
# net = MLPRegressor(hidden_layer_sizes=(7,), max_iter=1000, learning_rate_init=0.1, tol=1e-8)
# net.fit(input_train_norm, output_train_norm)
#
# # 预测数据
# m = 10  # 预测m个位置
# X_test = np.random.rand(m) * 10
# Y_test = np.random.rand(m) * 10
# tAB_test = (np.sqrt((A[0] - X_test) ** 2 + (A[1] - Y_test) ** 2) - np.sqrt((B[0] - X_test) ** 2 + (B[1] - Y_test) ** 2))
# tBC_test = (np.sqrt((B[0] - X_test) ** 2 + (B[1] - Y_test) ** 2) - np.sqrt((C[0] - X_test) ** 2 + (C[1] - Y_test) ** 2))
# input_test = np.vstack((tAB_test, tBC_test)).T
# real_locate = np.vstack((X_test, Y_test)).T
#
# # 预测数据归一化
# input_test_norm = scaler_input.transform(input_test)
#
# # BP网络预测输出
# BPoutput_norm = net.predict(input_test_norm)
#
# # 网络输出反归一化
# BPoutput = scaler_output.inverse_transform(BPoutput_norm)
#
# # 结果分析
# for i in range(m):
#     print('第{}次测试的实际位置是：({}, {})'.format(i+1, real_locate[i, 0], real_locate[i, 1]))
#     print('BP神经网络预测位置是：({}, {})'.format(BPoutput[i, 0], BPoutput[i, 1]))
#
# # 画图
# plt.scatter(real_locate[:, 0], real_locate[:, 1], marker='*', label='实际位置')
# plt.scatter(BPoutput[:, 0], BPoutput[:, 1], marker='o', label='预测位置')
# plt.legend()
# plt.title('BP网络预测输出')
# plt.xlabel('X方向')
# plt.ylabel('Y方向')
# plt.show()
#
# # 实际情况
# tAB_real = float(input('输入时间差1：'))
# tBC_real = float(input('输入时间差2：'))
# input_real = np.array([tAB_real, tBC_real]).reshape(1, -1)
# input_real_norm = scaler_input.transform(input_real)
# BPoutput_real_norm = net.predict(input_real_norm)
# BPoutput_real = scaler_output.inverse_transform(BPoutput_real_norm)
# print('预测位置是：({}, {})'.format(BPoutput_real[0, 0], BPoutput_real[0, 1]))



import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import os

path="E:/PycharmProjects/dian/data/"
files=os.listdir(path)


data_ori = np.zeros(shape=900,)
i = 0
for file in files:
    position=path+file
    a = np.loadtxt(position)
    for j in range(5):
        data_ori[j * 5 + i * 25] = a[(j + 1) * 9 - 8]  # 麦克风1峰值
        data_ori[j * 5 + 1 + i * 25] = a[(j + 1) * 9 - 5] # 麦克风2峰值
        data_ori[j * 5 + 2 + i * 25] = a[(j + 1) * 9 - 2] # 麦克风3峰值
        data_ori[j * 5 + 3 + i * 25] = a[(j + 1) * 9 - 7] - a[(j + 1) * 9 - 4]  # 时间：麦克风1-麦克风2
        data_ori[j * 5 + 4 + i * 25] = a[(j + 1) * 9 - 4] - a[(j + 1) * 9 - 1]  # 时间：麦克风2-麦克风3
    i = i + 1

data_ori = data_ori.reshape(-1,5)
X_train_t = data_ori
print(X_train_t.shape)

y_train_t = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                      [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                      [0, 2], [0, 2], [0, 2], [0, 2], [0, 2],
                      [0, 3], [0, 3], [0, 3], [0, 3], [0, 3],
                      [0, 4], [0, 4], [0, 4], [0, 4], [0, 4],
                      [0, 5], [0, 5], [0, 5], [0, 5], [0, 5],
                      [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                      [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                      [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
                      [1, 3], [1, 3], [1, 3], [1, 3], [1, 3],
                      [1, 4], [1, 4], [1, 4], [1, 4], [1, 4],
                      [1, 5], [1, 5], [1, 5], [1, 5], [1, 5],
                      [2, 0], [2, 0], [2, 0], [2, 0], [2, 0],
                      [2, 1], [2, 1], [2, 1], [2, 1], [2, 1],
                      [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                      [2, 3], [2, 3], [2, 3], [2, 3], [2, 3],
                      [2, 4], [2, 4], [2, 4], [2, 4], [2, 4],
                      [2, 5], [2, 5], [2, 5], [2, 5], [2, 5],
                      [3, 0], [3, 0], [3, 0], [3, 0], [3, 0],
                      [3, 1], [3, 1], [3, 1], [3, 1], [3, 1],
                      [3, 2], [3, 2], [3, 2], [3, 2], [3, 2],
                      [3, 3], [3, 3], [3, 3], [3, 3], [3, 3],
                      [3, 4], [3, 4], [3, 4], [3, 4], [3, 4],
                      [3, 5], [3, 5], [3, 5], [3, 5], [3, 5],
                      [4, 0], [4, 0], [4, 0], [4, 0], [4, 0],
                      [4, 1], [4, 1], [4, 1], [4, 1], [4, 1],
                      [4, 2], [4, 2], [4, 2], [4, 2], [4, 2],
                      [4, 3], [4, 3], [4, 3], [4, 3], [4, 3],
                      [4, 4], [4, 4], [4, 4], [4, 4], [4, 4],
                      [4, 5], [4, 5], [4, 5], [4, 5], [4, 5],
                      [5, 0], [5, 0], [5, 0], [5, 0], [5, 0],
                      [5, 1], [5, 1], [5, 1], [5, 1], [5, 1],
                      [5, 2], [5, 2], [5, 2], [5, 2], [5, 2],
                      [5, 3], [5, 3], [5, 3], [5, 3], [5, 3],
                      [5, 4], [5, 4], [5, 4], [5, 4], [5, 4],
                      [5, 5], [5, 5], [5, 5], [5, 5], [5, 5],
                      ])
print(y_train_t.shape)

# X_val_t = np.array([[4,2,8],
#                 [4,2,2],
#                 [5,1,4],
#                 [8,0,2]])
#
#
# y_val_t = np.array([[0,3],[4,5],[0,2],[2,3]])

X_train_t = torch.from_numpy(X_train_t).to(torch.float32)
y_train_t = torch.from_numpy(y_train_t).to(torch.float32)
# X_val_t = torch.from_numpy(X_val_t).to(torch.float32)
# y_val_t = torch.from_numpy(y_val_t).to(torch.float32)

def set_seed(seed=0):
    """Set one seed for reproducibility."""
    torch.backends.cudnn.deterministic = True  # 确定性设为真
    torch.backends.cudnn.benchmark = False  # 取消卷积算法加速
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 设置相同的种子，方便重复实验


def get_device():
    """Get a gpu if available."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


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


set_seed(0)
device = get_device()
print('using {} device'.format(device))
# 创建网络实例
classification_layers_x = [5, 16, 16, 1]
classification_layers_y = [5, 16, 16, 1]

cl_net = Network(classification_layers_x, classification_layers_y, 5, 1)
cl_net.to(device)
dataset_train = torch.utils.data.TensorDataset(X_train_t, y_train_t)  # 训练数据集, 把预处理好的训练数据集和标签放进来就行
# dataset_val = torch.utils.data.TensorDataset(X_val_t, y_val_t)  # 验证数据集 ，同上
train_DataLoader = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
# val_DataLoader = torch.utils.data.DataLoader(dataset_val, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
optim = torch.optim.Adam(cl_net.parameters(), lr=0.01, weight_decay=0.01)
loss1 = torch.nn.MSELoss()
loss1.to(device)
savefile = ''

# 开始训练
train_step = 0
val_step = 0

# writer = SummaryWriter('logs')
# trans = transforms.Compose()
epochs = 2000

for epoch in range(epochs):
    cl_net.train()
    train_loss = 0
    # 训练
    for Data in train_DataLoader:
        # 获取数据加载到GPU上
        data, label = Data
        data = data.to(device)
        label = label.to(device)
        # 梯度清零
        optim.zero_grad()
        # 前向传播
        output1, output2 = cl_net(data)
        # output1 = torch.reshape(output1, (2, 1))
        # output2 = torch.reshape(output2, (2, 1))
        set = torch.cat((output1, output2), axis=1)
        # 计算损失函数
        loss = loss1(set, label)
        # 反向传播计算梯度
        loss.backward()
        # 优化器使用梯度信息优化
        optim.step()

        # print('batch:{0},train_loss_step:{1}'.format(train_step, loss))
        train_step += 1
        train_loss = loss + train_loss
    print('epoch: {},train_loss: {:.10f}'.format(epoch, train_loss / train_step))
    # writer.add_scalar('train loss', train_loss / train_step, epoch)

# 储存
torch.save(cl_net, 'E:/PycharmProjects/dian/model.pth')
#
# # 加载
# cl_net = torch.load('C:/Users/12869/Desktop/model.pth')

# 验证
# cl_net.eval()
# val_predict = 0
# with torch.no_grad():
#         for Data in val_DataLoader:
#             data, label = Data
#             data = data.to(device)
#             label = label.to(device)
#             optim.zero_grad()
#
#             output_val_1, output_val_2 = cl_net(data)
#             output_val_1 = torch.reshape(output_val_1, (2, 1))
#             output_val_2 = torch.reshape(output_val_2, (2, 1))
#             set = torch.cat((output_val_1, output_val_2), axis=1)
#             # print(set, label)
#             # 计算损失函数
#             predict_step_val = loss1(set, label)
#             # print('验证损失:',predict_step_val.item())


# 测试
data_real= np.array([[1910,328,437,-77,0]])
data_real = torch.from_numpy(data_real).to(torch.float32)

x_real, y_real = cl_net(data_real)
print('预测值',torch.round(x_real).item(),torch.round(y_real).item())