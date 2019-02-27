import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch import autograd
import torch


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
num = 1000
for epoch in range(num):
    inputs = autograd.Variable(x_train)
    target = autograd.Variable(y_train)

    out = model(inputs)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backend()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num, loss.item()))