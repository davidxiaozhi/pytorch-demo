import torch
import torch.nn as nn
from torch.nn import Sequential
class Lenet(nn.Module):
    """
        Lenet 很简单的网络结构 没有使用任何激活函数
        Lenet(
          (layer1): Sequential(
            (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
            (maxool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          )
          (layer2): Sequential(
            (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
            (maxool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
          )
          (layer3): Sequential(
            (fc1): Linear(in_features=400, out_features=120, bias=True)
            (fc2): Linear(in_features=120, out_features=84, bias=True)
            (fc3): Linear(in_features=84, out_features=10, bias=True)
          )
        )

    """
    def __init__(self):
        super(Lenet, self).__init__()
        #第一次卷积 conv1 input_channel 1 output_channel 6 卷积核大小 3*3
        #第一次池化 池化内核 2*2 步长 2
        layer1 = Sequential()
        layer1.add_module("conv1", nn.Conv2d(1, 6, 3))
        layer1.add_module("maxool1", nn.MaxPool2d(2, 2))
        self.layer1 = layer1
        # 第二次卷积 conv1 input_channel 6 output_channel 16 卷积核大小 5*5
        # 第二次池化 池化内核 2*2 步长 2
        layer2 = Sequential()
        layer2.add_module("conv2",nn.Conv2d(6, 16, 5))
        layer2.add_module("maxool1", nn.MaxPool2d(2, 2))
        self.layer2 = layer2
        #接下来是三层全连接
        layer3 = Sequential()
        layer3.add_module("fc1", nn.Linear(400, 120))
        layer3.add_module("fc2", nn.Linear(120, 84))
        layer3.add_module("fc3", nn.Linear(84, 10))
        self.layer3 = layer3

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x;


if __name__ == '__main__':
    model = Lenet()
    print(model)