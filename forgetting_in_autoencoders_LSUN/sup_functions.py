import torch
import torch.nn as nn

def test_classifier(classif, data_loader):
  total = 0
  correct = 0
  for test_X, test_Y in data_loader:
    input_test = test_X.float().cuda()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    #cmat = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    #print(cmat)
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
    
  return correct/total*100

def test_classifier_on_generator(classif, gen_model, data_loader):
  total = 0
  correct = 0
  gen_model.eval()
  for idx, (test_X,  test_Y) in enumerate(data_loader):
    input_test = gen_model(test_X.float().cuda())
    outputs = classif(input_test.data)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    #cmat = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    #print(cmat)
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  gen_model.train()
  return correct/total*100

def get_indices_for_classes(data, data_classes):
  # Creates a list of indices of samples from the dataset, corresponding to given classes
  indices = torch.FloatTensor(list((data.tensors[1].long()==class_).tolist() for class_ in data_classes)).sum(0).nonzero().long().squeeze()
  return indices

def MSEloss_weighted(outputs, targets, coefficients, cuda = True):
  MSEloss = nn.MSELoss(reduction='none')
  MSEloss.cuda()
  coefficients = coefficients.cpu().tolist()
  coefficients_inv = [1/(a+1) for a in coefficients]
  return (MSEloss(outputs, targets).mean(1)*torch.Tensor(coefficients_inv).cuda()).sum()/sum(coefficients_inv)

def CrossEntropy_loss_weighted(outputs, targets, coefficients, cuda = True):
  #TODO implement if needed....
  return False
