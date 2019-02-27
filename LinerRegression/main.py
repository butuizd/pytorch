import models
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, autograd

#训练数据
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 确定损失函数和算法
model = models.LinerRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


num = 8000
for epoch in range(num):
   inputs = autograd.Variable(x_train)
   outputs = autograd.Variable(y_train)

   #forward ???
   out = model(inputs)
   loss = criterion(out, outputs) #??
   #backward
   optimizer.zero_grad()
   loss.backward()
    #update parameters
   optimizer.step()

   if (epoch + 1) % 100 == 0:
       print('epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num, loss.item()))

#
# model.eval()
# predict = model(autograd.Variable(x_train))
# predict = predict.data.numpy()
# plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label = 'original data')
# plt.plot(x_train.numpy(), predict, label = 'predict data')
# plt.show()


