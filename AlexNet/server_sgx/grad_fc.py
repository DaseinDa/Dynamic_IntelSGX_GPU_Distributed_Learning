import torch
import numpy as np
import os 
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
lr=0.01

input_npy=np.load("ServerSGX/global_memory/input.npy",allow_pickle=False)
weight_npy = np.load("ServerSGX/global_memory/weight.npy",allow_pickle=False)
outerror = np.load("ServerSGX/global_memory/outerror.npy",allow_pickle=False)
bias_npy = np.load("ServerSGX/global_memory/bias.npy",allow_pickle=False)


start = time.time()
weight_npy = torch.tensor(weight_npy).cuda()
outerror = torch.tensor(outerror).cuda()
input_npy = torch.tensor(input_npy).cuda()
bias_npy = torch.tensor(bias_npy).cuda()

inerror = torch.matmul(outerror, weight_npy.T).cuda()
wd=torch.tensor(np.zeros((weight_npy.shape[0],weight_npy.shape[1]))).cuda()
# for b in range(input_npy.shape[0]):
#     for i in range(weight_npy.shape[0]):
#         for j in range(weight_npy.shape[1]):
#             wd[i][j] += input_npy[b][i]*outerror[b][j]
input_npy = input_npy.transpose(0,1)
wd = torch.matmul(input_npy,outerror).cuda()
bd = outerror.sum(dim=0)
input_npy = input_npy.transpose(0,1)
inerror = inerror.reshape(input_npy.shape)

# print(input_npy.shape)
# print(weight_npy.shape)
# print(wd.shape)
# print(inerror.shape)

weight_npy = weight_npy+lr*wd
bias_npy = bias_npy+ lr*bd
torch.cuda.synchronize()
end = time.time()
print("Time execution grad_fc on GPU is:{}".format((end-start)*1000))
np.save("ServerSGX/global_memory/inerror.npy",np.float64(inerror.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/weight.npy",np.float64(weight_npy.cpu().detach().numpy()))
np.save("ServerSGX/global_memory/bias.npy",np.float64(bias_npy.cpu().detach().numpy()))