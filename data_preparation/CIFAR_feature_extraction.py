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

cuda = torch.device('cuda')

opt = {
      'im_size': 32,
      'images_per_class_train': 100000,
      'images_per_class_test': 10000,
      'batch_size': 20,
      'batch_number': 5500,
      'CIFAR_version': 100,
     }

print("Loading data ...")
transform_train = transforms.Compose([                                                                    
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])
transform_test = transforms.Compose([                                                                    
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])

#
project_root = '/home/besedin/workspace/Projects/Journal_paper/'
if opt['CIFAR_version']==10:
  trainset = torchvision.datasets.CIFAR10(root=project_root+'datasets/original_data/CIFAR/', train=True, download=True, transform=transform_train) 
  testset = torchvision.datasets.CIFAR10(root=project_root+'datasets/original_data/CIFAR/', train=False, download=True, transform=transform_test) 
elif opt['CIFAR_version']==100:
  trainset = torchvision.datasets.CIFAR100(root=project_root+'datasets/original_data/CIFAR/', train=True, download=True, transform=transform_train) 
  testset = torchvision.datasets.CIFAR100(root=project_root+'datasets/original_data/CIFAR/', train=False, download=True, transform=transform_test) 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt['batch_size'], shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt['batch_size'], shuffle=False, num_workers=2)
#original_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



print('Loading model')
resnet152 =  models.resnet152(pretrained=True)
resnet = nn.Sequential(*list(resnet152.children())[:-2]).cuda()
del resnet152


trainset_features = (torch.zeros(50000, 2048), torch.zeros(50000))
testset_features =  (torch.zeros(10000, 2048), torch.zeros(10000))

  # Choose the data folder with the lmdb folders
for batch_idx, (inputs, targets) in enumerate(trainloader):
  print(batch_idx)
  out = resnet(inputs.data.cuda())
  trainset_features[0][batch_idx*opt['batch_size']:(batch_idx+1)*opt['batch_size']] = out.squeeze().float().data  
  trainset_features[1][batch_idx*opt['batch_size']:(batch_idx+1)*opt['batch_size']] = targets
  
for batch_idx, (inputs, targets) in enumerate(testloader):
  print(batch_idx)
  out = resnet(inputs.data.cuda())
  testset_features[0][batch_idx*opt['batch_size']:(batch_idx+1)*opt['batch_size']] = out.squeeze().float().data
  testset_features[1][batch_idx*opt['batch_size']:(batch_idx+1)*opt['batch_size']] = targets
  
torch.save(trainset_features, '/home/besedin/workspace/Projects/Journal_paper/datasets/CIFAR'+str(opt['CIFAR_version'])+'_features/trainset.pt')
torch.save(testset_features, '/home/besedin/workspace/Projects/Journal_paper/datasets/CIFAR'+str(opt['CIFAR_version'])+'_features/testset.pt')
