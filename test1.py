import torch
import numpy as np
import pointops
from pointops import farthest_point_sampling

data = torch.load('data/tgnet_resize_dataset/train/0EAKT1CU_lower.pth')
coord = data['coord']
print(type(coord))
coord = torch.from_numpy(coord).to(torch.float).cuda(2)
print(coord.shape)
offset = torch.tensor([30000,60000]).to(torch.int32).cuda(2)
new_offset = torch.tensor([3000,5000]).to(torch.int32).cuda(2)
offset = torch.cuda.IntTensor(offset)
new_offset = torch.cuda.IntTensor(new_offset)

idx = farthest_point_sampling(coord,offset,new_offset).cuda(2)
# print(offset,new_offset)
print(type(idx),idx.device,idx.shape)
print(coord)
print(idx.long())
idx = 1

new_coord = coord[idx,:3]
new_coord += 1
# print(new_coord)
# print(idx.long())