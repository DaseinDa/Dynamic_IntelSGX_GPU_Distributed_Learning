import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import time
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import pickle

# #input(batchsize, inchannel, width, height)
# input = torch.zeros(100,3,28,28)
# #weight(outchannel,inchannel,filer_width,filter_height)
# weight = nn.Parameter(torch.randn(16, 3, 5, 5))  # 自定义的权值
# bias = nn.Parameter(torch.zeros(6,dtype=torch.float64))

data_npy=np.load("ServerSGX/global_memory/data.npy",allow_pickle=False)
weight_npy=np.load("ServerSGX/global_memory/weight.npy",allow_pickle=False)
bias_npy = np.load("ServerSGX/global_memory/bias.npy")
torch.cuda.synchronize()


# print(data_npy)
# print(data_npy.shape)
##### 数据放到GPU上
start = time.time()
data_npy = torch.tensor(data_npy).cuda()
weight_npy = torch.tensor(weight_npy).cuda()
bias_npy = torch.tensor(bias_npy).cuda()
###### 交换weight第一和第二维度#########
weight_npy = weight_npy.transpose(0,1)

conv = torch.nn.functional.conv2d(data_npy,weight_npy,bias_npy,stride=1,padding=0)
torch.cuda.synchronize()
end = time.time()

print("Time execution conv on GPU is:{}".format((end-start)*1000))
# print(conv)

print(data_npy.shape)
print(weight_npy.shape)
np.save("ServerSGX/global_memory/data.npy",np.float64(conv.cpu().detach().numpy()))
# print("Hello again\n")
# print("Here is test.py\n")