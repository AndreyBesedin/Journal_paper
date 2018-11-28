import torch
import torch.nn as nn
import torch.nn.functional as F
from progress.bar import Bar
from torch.utils.data import DataLoader

def test_model(classif_, data_loader_):
  total = 0
  correct = 0
  for idx, (test_X, test_Y) in enumerate(data_loader_):
    input_test = test_X.cuda()
    outputs = classif_(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def test_model_on_gen(classif_, autoenc, data_loader_):
  total = 0
  correct = 0
  for idx, (test_X,  test_Y) in enumerate(data_loader_):
    input_test = autoenc(test_X.cuda())
    outputs = classif_(input_test)
#    outputs = classif_(test_X.cuda())
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.kaiming_normal_(m.weight.data)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
  elif classname.find('Linear')!= -1:
    torch.nn.init.kaiming_normal_(m.weight.data)
    m.bias.data.fill_(0)

def reconstruct_dataset_with_AE(dataset, rec_model, bs = 100, real_data_ratio=0):
  data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
  data_size = dataset.tensors[0].shape[0]
  res_data = torch.zeros(dataset.tensors[0].shape)
  res_labels = torch.zeros(data_size)
  current_index = 0
  bar = Bar('Reconstructing data for absent classes:', max=int(data_size/bs))
  for idx, (train_x, train_y) in enumerate(data_loader):
      #call('nvidia-smi')
    bar.next()
    inputs = train_x.cuda()
    batch = {}
    if idx < real_data_ratio:
      batch = inputs
    else:
      batch = rec_model(inputs)
    current_batch_size = batch.shape[0]
    res_data[current_index:current_index+current_batch_size] = batch.cpu().data
    res_labels[current_index:current_index+current_batch_size] = train_y
    current_index+=current_batch_size 
  bar.finish()
  return (res_data, res_labels)
