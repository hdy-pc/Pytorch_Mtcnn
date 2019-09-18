import torch
import numpy as np
cond = torch.tensor(np.array([[1],[0],[1]]))
a=torch.tensor(np.array([[1,2,3,1],[0,0,0,0],[3,3,3,3]]))
m=np.array([1,2,3])
n = np.array([1,2,3])
print(m*n)
print(a[:,:2])
# b = cond>0
# print(b)
# index= torch.nonzero(b)[:,0]
# print(index)
# z=a[index]
# # z=torch.masked_select(a,index)
# print(z)
# cond=np.arange(12).reshape(1,1,3,4)[0][0]
# map = torch.tensor(cond)
# print(map)
# # index =np.where(map<5)
# index = torch.nonzero(torch.gt(map,5))
# print(index)#(array([0, 0, 0, 0, 1], dtype=int64), array([0, 1, 2, 3, 0], dtype=int64), array([0, 0, 0, 0, 0], dtype=int64))
# offset = np.arange(24).reshape(4,2,3)
# map = np.array([[1,2,3],[4,5,6]])
# indexs =np.array([[1,2],[1,1]])
# # print(offset)
# d_offset = offset[:, indexs[:, 0], indexs[:, 1]]
# cls = map[indexs[:,0],indexs[:,1]]
# # print(type(d_offset))
# # print(cls)
# a=np.array([1,2,3])
# b=np.array([4,5,6])
#
# c=np.array([a,b]).T
# # print(c)
#
# a=np.arange(12).reshape(12,1,1,1)
# print(a)
# print(np.where(a<6)[0])