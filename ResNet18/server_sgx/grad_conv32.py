import torch
import numpy as np
import time
# from torch.autograd.function import Function
# from torch.autograd import gradcheck
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
lr=0.01
torch.set_printoptions(threshold=10000)

input_npy=np.load("ServerSGX/global_memory/input.npy",allow_pickle=False)
weight_npy = np.load("ServerSGX/global_memory/weight.npy",allow_pickle=False)
outerror = np.load("ServerSGX/global_memory/outerror.npy",allow_pickle=False)
bias_npy = np.load("ServerSGX/global_memory/bias.npy",allow_pickle=False)

start = time.time()
outerror = torch.tensor(outerror).cuda()
input = torch.tensor(input_npy).cuda()
weight = torch.tensor(weight_npy).cuda()
bias = torch.tensor(bias_npy).cuda()
###### 交换weight第一和第二维度#########
weight = weight.transpose(0,1)
# grad_output = torch.tensor(np.ones((128,6,24,24)))
# input = torch.tensor(np.ones((128,1,28,28)))
# weight = torch.tensor(np.ones((6,1,5,5)))
# print(weight.shape)
# print(input.shape)
# print(outerror.shape)

# test_input = torch.tensor(np.ones((10,256,10,10)))
# test_oute = torch.tensor(np.ones((10,256,7,7)))
# test_w = torch.tensor(np.ones((256,256,3,3)))
# inerror = torch.nn.grad.conv2d_input(input.shape, weight,outerror,stride=1,padding=1).float()
# print(input.shape)
# print(weight.shape)
# print(outerror.shape)./main
inerror = torch.nn.grad.conv2d_input(input.shape, weight,outerror,stride=2,padding=1)
wd = torch.nn.grad.conv2d_weight(input, weight.shape, outerror,stride=2,padding=1)
bd = outerror.sum(dim=(0,2,3)).cuda()

weight = weight+lr*wd
bias = bias+ lr*bd
torch.cuda.synchronize()
end = time.time()

print("Time execution grad_conv 32 on GPU is:{}".format((end-start)*1000))

np.save("ServerSGX/global_memory/inerror.npy",np.float64(inerror.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/weight.npy",np.float64(weight.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/bias.npy",np.float64(bias.cpu().detach().numpy()))
torch.cuda.empty_cache()