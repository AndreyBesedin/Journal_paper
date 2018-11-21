import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import sampler
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from synthetic_data_generation import initialize_synthetic_sampler, sample_data_from_sampler
import os

root = '~/workspace/Projects/Journal_paper/'
dataset = 'synthetic'
print('Loading data')
opts = {
  'batch_size': 100,
  'mode': 'multi-class',
  'dataset': 'LSUN',
  'latent_space_size': 4,
  'learning_rate': 0.001,
  'number_of_epochs': 100,
  'add_noise': True,
  'dim': 2048,
  'nb_classes': 100,
  'predefined_sampler': True,
  'code_size': 32,
  'betta': 0.2,
  }
  
def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 2048)
  return x

data_sampler = torch.load('./models/data_sampler_'+ str(opts['nb_classes']) +'_classes.pth')
if not opts['predefined_sampler']:
  data_sampler = initialize_synthetic_sampler(opts['dim'], opts['nb_classes'], 1.7)

ls = opts['code_size']
train_class_size = 1000
test_class_size = 1000

trainset_ = sample_data_from_sampler(data_sampler, train_class_size)
testset_ = sample_data_from_sampler(data_sampler, test_class_size)

trainset = data_utils.TensorDataset(trainset_[0], trainset_[1])
testset = data_utils.TensorDataset(testset_[0], testset_[1])
train_loader = data_utils.DataLoader(trainset, batch_size=100, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=100, shuffle = False)

class autoencoder(nn.Module):
  def __init__(self):
    global ls
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(2048, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(True), 
      nn.Linear(32, ls),
    )
    self.decoder = nn.Sequential(
      nn.Linear(ls, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(True),
      nn.Linear(32, 128),  
      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 512),  
      nn.BatchNorm1d(512),
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


autoencoder_model = autoencoder().cuda()
classifier = torch.load('./models/batch_classifier_'+ str(nb_classes) +'_classes.pth')

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
criterion_classif = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['learning_rate'], betas=(0.5, 0.999),
                             weight_decay=1e-5)
optimizer_class = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['betta']*opts['learning_rate'],
                             weight_decay=1e-5)

acc = test_model(classifier, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X
    labels = train_Y.cuda()
    img = Variable(inputs).cuda()
    orig_classes = classifier(img)
#    img = Variable(inputs).cuda()
    # ===================forward=====================
    output = autoencoder_model(img)
    classification_reconstructed = classifier(output)
    loss_classif = criterion_classif(classification_reconstructed, orig_classes)
    optimizer.zero_grad()
    loss_classif.backward(retain_graph=True)
    optimizer_class.step()
    loss = criterion(output, img)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ===================log========================
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.data[0]))
  if epoch % 1 == 0:
    autoencoder_model.eval()
    acc = test_model_on_gen(classifier, autoencoder_model, test_loader)
    autoencoder_model.train()
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
