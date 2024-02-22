import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time


input = torch.randn(batchsize=100,channel=3,height=28,width=28)
weight = nn.Parameter(torch.randn(16, 1, 5, 5))  # 自定义的权值
bias = nn.Parameter(torch.randn(16)) 


conv = torch.nn.functinoal.Conv2d(input,weight,bias,stride=1,padding=0)
print("Hello again\n")
print("Here is test.py\n")