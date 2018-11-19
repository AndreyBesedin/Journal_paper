import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.fc1  = nn.Linear(64*28*28, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 28,28)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x.view(batch_size, 64*28*28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def incremental_stream(nb_classes, rate, duration, vect=None):
  if vect==None:
    vect = (torch.ones(nb_classes)/nb_classes)
  else:
    vect/=vect.sum()
  for interval in range(duration):
    S = itertools.accumulate(vect)
    q = np.random.uniform(0, 1)
    for idx, s in enumerate(S):
      if s > q:
        vect[idx]/=rate
        vect/=vect.sum()
        yield idx
        break

def unordered_stream(nb_classes, rate, duration, classes_per_interval=5, vect=None):
  if vect==None:
    vect = (torch.ones(nb_classes)/nb_classes)
  else:
    vect/=vect.sum()
  for interval in range(duration):
    res_classes = torch.zeros(classes_per_interval)
    Q = np.random.uniform(0, 1, classes_per_interval)
    for idx_q, q in enumerate(Q):
      S = itertools.accumulate(vect)
      for idx_s, s in enumerate(S):
        if s > q:
          vect[idx_s]/=rate
          vect/=vect.sum()
          res_classes[idx_q] = idx_s
          break
    yield res_classes

def test_model(model, test_loader):
  total = 0
  correct = 0
  conf_matrix = np.zeros((10,10))
  for idx, (test_x, test_y) in enumerate(test_loader):
    input_test = test_x.cuda()
    outputs = model(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
    conf_matrix+=confusion_matrix(predicted.cpu().numpy(), labels.numpy())
  
  return correct/total*100, (conf_matrix/conf_matrix.sum(0)*100).transpose(0,1)

def main():
  model = Net().cuda()

  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
  max_test_acc = 0

  img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  root = '/home/besedin/workspace/Projects/Journal_paper/'
  train_dataset = datasets.MNIST(root=root+'datasets/',
    train=True,
    download=True,
    transform=img_transform)
  test_dataset = datasets.MNIST(root=root+'datasets/',
    train=False,
    download=True,
    transform=img_transform)

  test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
  test_acc = test_model(model, test_loader)

  #train_dataset_ = torch.load('../datasets/LSUN_features/testset.pt')
  #train_dataset = TensorDataset(train_dataset_[0], train_dataset_[1])
  intervals = incremental_stream(10, 10, 1000)
  for idx_interval, classes in enumerate(intervals):
    data_class = classes
    print(classes)
    indices = (train_dataset.train_labels.long()==data_class).nonzero().long()
    train_loader = DataLoader(train_dataset, batch_size=1000, sampler=SubsetRandomSampler(indices.squeeze()))
    running_loss = 0.0
    for idx, (train_x, train_y) in enumerate(train_loader):
      inputs = train_x.cuda()
      labels = train_y.cuda()
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels.long())
      loss.backward()
      optimizer.step()

        # print statistics
      running_loss += loss.item()
      if idx % 100 == 99:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, idx_interval + 1, running_loss / 100))
        running_loss = 0.0
        
      test_acc, confusion = test_model(model, test_loader)
      if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_model = model.float()
      #torch.save(best_model, 'models/'+ dataset +'_classifier.pt')
    print(confusion)
    print('Test accuracy: ' + str(test_acc))

if __name__ == '__main__':
  main()
