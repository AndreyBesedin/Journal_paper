import torch
import torch.nn as nn
import torch.nn.functional as F

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

