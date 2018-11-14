import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os

root = '/home/besedin/workspace/Projects/Journal_paper/'
dataset = 'LSUN'
print('Loading data')
opts = {
  'batch_size': 100,
  'mode': 'multi-class',
  'dataset': 'LSUN',
  'latent_space_size': 4,
  'learning_rate': 0.001,
  'number_of_epochs': 200,
  }
  
def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 2048)
  return x

train_dataset_ = torch.load(root + 'datasets/LSUN_features/testset.pt')
#max_train = max(train_dataset_[0].max(), -train_dataset_[0].min())
#train_data = train_dataset_[0]/max_train
train_dataset = TensorDataset(train_dataset_[0], train_dataset_[1])
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle = True)

#mean_ = trainset[0].mean(); std_ = trainset[0].std()
test_dataset_ = torch.load(root + 'datasets/LSUN_features/testset.pt')
#test_data = test_dataset_[0]/max_train
test_dataset = TensorDataset(test_dataset_[0], test_dataset_[1])
test_loader = DataLoader(test_dataset, batch_size=100, shuffle = False)
#train_dataset = ((train_dataset[0] - mean_)/std_, train_dataset[1])
#test_dataset = ((test_dataset[0] - mean_)/std_, test_dataset[1])

#batch = train_dataset.train_data[0:100].float()

class autoencoder(nn.Module):
  def __init__(self):
    ls = 128
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(2048, 512),
#      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 128),
#      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 128),
#      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, ls),
    )
    self.decoder = nn.Sequential(
      nn.Linear(ls, 128),
#      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 128),  
#      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 512),
#      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 2048),
#      nn.ReLU(True)
    )

  def forward(self, x):
    batch_size = x.size(0)
 #   x = x.view(batch_size, 1, 28,28)
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, 30)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x


autoencoder = autoencoder().cuda()
classifier = torch.load(root + 'batch_training/results/LSUN/models/LSUN_classifier_original.pt')

def test_model_on_gen(classif, autoenc, test_loader):
  total = 0
  correct = 0
  for idx, (test_X,  test_Y) in enumerate(test_loader):
    input_test =autoenc(test_X.cuda())
    outputs = classif(input_test)
#    outputs = classif(test_X.cuda())
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def test_model(classif, test_loader):
  total = 0
  correct = 0
  for idx, (test_X, test_Y) in enumerate(test_loader):
    input_test = test_X.cuda()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=opts['learning_rate'],
                             weight_decay=1e-7)

acc = test_model(classifier, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X
    labels = train_Y.cuda()
    img = Variable(inputs).cuda()
    # ===================forward=====================
    output = autoencoder(img)
    loss = criterion(output, img)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ===================log========================
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.data[0]))
  if epoch % 1 == 0:
    autoencoder.eval()
    acc = test_model_on_gen(classifier, autoencoder, test_loader)
    autoencoder.train()
    #img_vis = train_dataset[0][0:bs]
    #img = Variable(img_vis).cuda()
    #output = autoencoder(img)
    #out = torch.cat((img, output), 0)
    #pic = to_img(out.cpu().data)
    #save_image(pic, './temp_images/image_{}.png'.format(epoch))
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
