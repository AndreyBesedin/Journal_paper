print('Loading dependancies')
import six
import lmdb
import torch
import cv2
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

original_classes = ['airplane', 'bedroom', 'bicycle', 'bird', 'boat', 
                    'bottle', 'bridge', 'bus', 'car', 'cat', 
                    'chair', 'church_outdoor', 'classroom', 'conference_room', 'cow',
                    'dining_room', 'dining_table', 'dog', 'horse', 'kitchen', 
                    'living_room', 'motorbike', 'person', 'potted_plant', 'restaurant', 
                    'sheep', 'sofa', 'tower', 'train', 'tv-monitor']

opt = {
      'im_size': 224,
      'images_per_class_train': 100000,
      'images_per_class_test': 10000,
      'batch_size': 20,
      'batch_number': 5500,
     }

class Identity(nn.Module):                                                                    
  def __init__(self):                                     
     super(Identity, self).__init__()                                                                                                           
  def forward(self, x):                                                              
    return x  

print('Loading model')
resnet152 =  models.resnet152(pretrained=True)
resnet = nn.Sequential(*list(resnet152.children())[:-1]).cuda()
del resnet152

for idx in range(3): resnet[7][idx].relu = Identity()

print('loading data')
cuda = torch.device('cuda')
transform = transforms.Compose([
  transforms.RandomCrop(opt['im_size']),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
nb_classes = len(original_classes)
#trainset = torch.zeros()

for idx_class in range(1):
  res = torch.zeros(110000, 2048)
  data_class = 'tv-monitor'
  # Choose the data folder with the lmdb folders
  lmdb_dir = '/home/besedin/workspace/Data/LSUN/data_lmdb/train/'+data_class+'_train_lmdb/'
#  lmdb_dir = '/data/LSUN/'+data_class+'_train_lmdb/'
  bs = 20
  count=0
  batch = torch.zeros(bs, 3, 224, 224).cuda()
  print('extracting features')
  with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
    for idx, (key, value) in enumerate(txn.cursor()):
      if idx!=1e+10:
        print(idx)
        img_numpy = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
        img_pil = Image.fromarray(img_numpy.astype('uint8'), 'RGB')
        img_tensor = transform(img_pil).reshape(1,3,224, 224).cuda()
        batch[count%bs] = img_tensor
        count+=1
      if count%bs==0:
        out = resnet(batch.data)
        res[count-bs:count] = out.squeeze().float().data
        if count>=110000-1: break
  torch.save(res, '/home/besedin/workspace/Projects/Journal_paper/datasets/LSUN_features/by_class_with_names/'+data_class+'_train_test_no_relu.pt')
