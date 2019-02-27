from torch import nn


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)   # nn.Liner是一种线性结构  需要设置输入和输出的节点数

    #处理传入数据
    def forward(self, x):
        out = self.linear(x)
        return out
