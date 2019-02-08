import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 30
opts = {
  'batch_size': 1000,
  'learning_rate': 0.001
  }
  
class Classifier(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(2048, 784)
    self.fc2 = nn.Linear(784, 256)
    self.fc3 = nn.Linear(256, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def test_classifier(classif, data_loader):
  total = 0
  correct = 0
  for test_X, test_Y in data_loader:
    input_test = test_X.float().cuda()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def get_indices_for_classes(data, data_classes):
  # Creates a list of indices of samples from the dataset, corresponding to given classes
  indices = torch.FloatTensor(list((data.tensors[1].long()==class_).tolist() for class_ in data_classes)).sum(0).nonzero().long().squeeze()
  return indices

# Loading the datasets
trainset = torch.load('../datasets/LSUN/trainset.pth')
testset = torch.load('../datasets/LSUN/testset.pth')
trainset = TensorDataset(trainset[0], trainset[1])
testset = TensorDataset(testset[0], testset[1])
pretrain_on_classes = range(15)
indices_train = get_indices_for_classes(trainset, pretrain_on_classes)
indices_test = get_indices_for_classes(testset, pretrain_on_classes)

# Initializing data loaders for first 5 classes
train_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_train))
test_loader = DataLoader(testset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_test))

# Initializing classification model, criterion and optimizer function
classifier = Classifier(nb_of_classes)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
#classification_criterion = nn.CrossEntropyLoss()
classification_criterion = nn.CrossEntropyLoss()

classifier.cuda()
classification_criterion.cuda()

max_accuracy = 0
training_epochs = 25
for epoch in range(training_epochs):
  # Training the classifer
  for idx, (X, Y) in enumerate(train_loader):
    inputs = X.cuda()
    labels = Y.long().cuda()
    outputs = classifier(inputs)
    classification_loss = classification_criterion(outputs, labels)
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    if idx%50==0:
      print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, training_epochs,  classification_loss.item()))
  acc = test_classifier(classifier, test_loader)
  print('Test accuracy after {} epochs: {:.8f}'.format(epoch+1, acc))    
  if acc > max_accuracy:
    max_accuracy = acc
    torch.save(classifier.state_dict(), './pretrained_models/classifier_15_LSUN.pth')
    
    
