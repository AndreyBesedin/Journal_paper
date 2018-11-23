import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler

root = '~/workspace/Projects/Journal_paper/'
dataset = 'MNIST'
print('Loading data')
bs = 600

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#img_transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=root+'datasets/',
  train=True,
  download=True,
  transform=img_transform) 
test_dataset = datasets.MNIST(root=root+'datasets/',
  train=False,
  download=True,
  transform=img_transform)
#train_dataset = datasets.EMNIST(root=root+'datasets/',
  #train=True,
  #split='balanced',
  #download=True,
  #transform=transforms.ToTensor()) 
#test_dataset = datasets.EMNIST(root=root+'datasets/',
  #train=False,
  #split='balanced',
  #download=True,
  #transform=transforms.ToTensor())
  
#indices = torch.randperm(60000)[:3000].long()
#train_loader = DataLoader(train_dataset, batch_size=100, sampler=SubsetRandomSampler(indices))
train_loader = DataLoader(train_dataset, shuffle=True,
  batch_size=bs)
test_loader = DataLoader(test_dataset, shuffle=False,
  batch_size=bs)

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

def test_model(model, test_loader):
  total = 0
  correct = 0
  for idx, (test_x, test_y) in enumerate(test_loader):
    input_test = test_x.cuda()
    outputs = model(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100
  
model = Net().cuda()
test_acc = test_model(model, test_loader)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
max_test_acc = 0

for epoch in range(25):  # loop over the dataset multiple times
    running_loss = 0.0
#    for idx, data in enumerate(trainloader, 0):
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
              (epoch + 1, idx + 1, running_loss / 100))
        running_loss = 0.0
        
    test_acc = test_model(model, test_loader)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = model.float()
      torch.save(best_model, 'models/'+ dataset +'_small_classifier.pth')
      
    print('Test accuracy: ' + str(test_acc))

print('Finished Training')
