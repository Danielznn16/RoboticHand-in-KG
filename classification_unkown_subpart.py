#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as pt
import importlib
import os
import sys
import numpy as np
from data_utils.ModelNetDataLoader import pc_normalize, ModelNetDataLoader


# In[2]:


model_name = 'pointnet2_cls_msg'
# device_name = 'cuda:0'
device_name = 'cpu'
num_class = 50


# In[3]:


# set pytorch
device = pt.device(device_name)


# In[4]:


# add env to import model
sys.path.append(os.path.join('./','models'))


# In[5]:


# load model
MODEL = importlib.import_module(model_name)
model = MODEL.get_model(num_class,normal_channel=True).to(device)

checkpoint = pt.load('./log/classification/pointnet2_cls_msg_part/checkpoints/best_model.pth',map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


# In[6]:


catfile = './data/partcls/modelnet40_shape_names.txt'
cat = [line.rstrip() for line in open(catfile)]
classes = dict(zip(cat, range(len(cat))))
to_class = [None]*len(classes)
for cls in classes:
    to_class[classes[cls]] = cls


# In[7]:


# point_set = np.loadtxt('./data/modelnet40_normal_resampled/chair/chair_0001.txt',delimiter=',').astype(np.float32)
# print(point_set.shape)


# In[8]:


classifier = model.eval()
def get_subpart_cls_result(point_set):
    with pt.no_grad():
        point_set = point_set[0:256,:]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        result = classifier(pt.Tensor([point_set]).to(device).transpose(2, 1))[0][0].data
        return result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




