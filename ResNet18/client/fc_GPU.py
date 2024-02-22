import torch
import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
data_npy=np.load("ServerSGX/global_memory/data.npy",allow_pickle=False)
weight_npy=np.load("ServerSGX/global_memory/weight.npy",allow_pickle=False)
bias_npy = np.load("ServerSGX/global_memory/bias.npy")

# fc_layer=torch.nn.Linear(3,4)
# fc_layer.weight.data = torch.tensor(np.zeros((4,3)))
# fc_layer.bias.data.fill_(0)
# inputs = torch.tensor(np.ones(3))
# output = fc_layer(inputs)

data_npy = torch.tensor(data_npy).cuda()
weight_npy = torch.tensor(weight_npy).cuda()
bias_npy = torch.tensor(bias_npy).cuda()

fc_layer=torch.nn.Linear(weight_npy.shape[0],weight_npy.shape[1])
fc_layer.weight.data = weight_npy.T
fc_layer.bias.data = bias_npy
print(data_npy.shape)
print(weight_npy.shape)
output = fc_layer(data_npy)

# print(output)
np.save("ServerSGX/global_memory/data.npy",np.float64(output.cpu().detach().numpy()))