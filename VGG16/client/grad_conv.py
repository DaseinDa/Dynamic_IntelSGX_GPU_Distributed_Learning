import torch
import numpy as np
# from torch.autograd.function import Function
# from torch.autograd import gradcheck
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
lr=0.01
torch.set_printoptions(threshold=10000)

input_npy=np.load("ServerSGX/global_memory/input.npy",allow_pickle=False)
weight_npy = np.load("ServerSGX/global_memory/weight.npy",allow_pickle=False)
outerror = np.load("ServerSGX/global_memory/outerror.npy",allow_pickle=False)
bias_npy = np.load("ServerSGX/global_memory/bias.npy",allow_pickle=False)


outerror = torch.tensor(outerror).cuda()
input = torch.tensor(input_npy).cuda()
weight = torch.tensor(weight_npy).cuda()
bias = torch.tensor(bias_npy).cuda()
###### 交换weight第一和第二维度#########
weight = weight.transpose(0,1)
# grad_output = torch.tensor(np.ones((128,6,24,24)))
# input = torch.tensor(np.ones((128,1,28,28)))
# weight = torch.tensor(np.ones((6,1,5,5)))
inerror = torch.nn.grad.conv2d_input(input.shape, weight, outerror).cuda()
wd = torch.nn.grad.conv2d_weight(input, weight.shape, outerror).cuda()
bd = outerror.sum(dim=(0,2,3)).cuda()

weight = weight+lr*wd
bias = bias+ lr*bd
print(wd.shape)
np.save("ServerSGX/global_memory/inerror.npy",np.float64(inerror.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/weight.npy",np.float64(weight.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/bias.npy",np.float64(bias.cpu().detach().numpy()))