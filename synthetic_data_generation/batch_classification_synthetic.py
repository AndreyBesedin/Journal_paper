import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 
from synthetic_data_generation import initialize_synthetic_sampler, sample_data_from_sampler
root = '/home/besedin/workspace/Projects/Journal_paper/'
dataset = 'synthetic'
print('Loading data')


dim = 2048
nb_classes = 10
train_class_size = 1000
test_class_size = 1000
data_sampler = initialize_synthetic_sampler(dim, nb_classes, 0.1)

trainset_ = sample_data_from_sampler(data_sampler, train_class_size)
testset_ = sample_data_from_sampler(data_sampler, test_class_size)

trainset = data_utils.TensorDataset(trainset_[0], trainset_[1])
testset = data_utils.TensorDataset(testset_[0], testset_[1])
train_loader = data_utils.DataLoader(trainset, batch_size=100, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=100, shuffle = False)
#mean_ = trainset[0].mean(); std_ = trainset[0].std()
#trainset = ((trainset[0] - mean_)/std_, trainset[1])
#testset = ((testset[0] - mean_)/std_, trainset[1])

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, 100)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

def test_model(model, testset):
  total = 0
  correct = 0
  test_size = len(testset[0])
  for idx, (test_X, test_Y) in enumerate(test_loader):
    input_test = test_X.cuda()
    outputs = model(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100
  
model = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
data_size = len(trainset[0])
max_test_acc = 0
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    indices = torch.randperm(data_size)
#    for idx, data in enumerate(trainloader, 0):
    for idx, (train_X, train_Y) in enumerate(train_loader):
      inputs = train_X.cuda()
      labels = train_Y.cuda()

        # zero the parameter gradients
      optimizer.zero_grad()

        # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels.long())
      loss.backward()
      optimizer.step()

        # print statistics
      running_loss += loss.item()
      if idx % 10 == 9:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, idx + 1, running_loss / 100))
        running_loss = 0.0
        
    test_acc = test_model(model, testset)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = model.float()
      #torch.save(best_model, './results/synthetic/models/'+dataset+'_classifier_original.pth')
      
    print('Test accuracy: ' + str(test_acc))


print('Test accuracy')
print('Finished Training')
