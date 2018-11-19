import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
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
  'number_of_epochs': 100,
  'add_noise': True,
  }
  
def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 2048)
  return x

train_dataset_ = torch.load(root + 'datasets/LSUN_features/trainset.pt')
#max_train = max(train_dataset_[0].max(), -train_dataset_[0].min())
#train_data = train_dataset_[0]/max_train
train_dataset = TensorDataset(train_dataset_[0], train_dataset_[1])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle = True)
#indices = torch.tensor(list((l in range(10) for l in train_dataset.tensors[1]))).nonzero().long()
#indices = torch.randperm(300000)[:100000]
#train_loader = DataLoader(train_dataset, batch_size=100, sampler=SubsetRandomSampler(indices.squeeze()))
#mean_ = trainset[0].mean(); std_ = trainset[0].std()
test_dataset_ = torch.load(root + 'datasets/LSUN_features/testset.pt')
#test_data = test_dataset_[0]/max_train
test_dataset = TensorDataset(test_dataset_[0], test_dataset_[1])
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle = False)
#test_loader = DataLoader(test_dataset, batch_size=1000, sampler=SubsetRandomSampler(indices.squeeze()))
#train_dataset = ((train_dataset[0] - mean_)/std_, train_dataset[1])
#test_dataset = ((test_dataset[0] - mean_)/std_, test_dataset[1])

#batch = train_dataset.train_data[0:100].float()

class autoencoder(nn.Module):
  def __init__(self):
    ls = 32
    dropout = 0
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(2048, 3072),
      nn.BatchNorm1d(3072),
      nn.ReLU(True),
      nn.Linear(3072, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(True),
      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(True),
      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(True), 
      nn.Linear(64, ls),
    )
    self.decoder = nn.Sequential(
      nn.Linear(ls, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(True),
      nn.Linear(64, 128),  
      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Linear(128, 256),  
      nn.BatchNorm1d(256),
      nn.ReLU(True),
      nn.Linear(256, 512),  
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 1024),  
      nn.BatchNorm1d(1024),
      nn.ReLU(True),
      nn.Linear(1024, 2048),
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
criterion_classif = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['learning_rate'], betas=(0.5, 0.999),
                             weight_decay=1e-5)
opts['betta'] = 1
optimizer_class = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['betta']*opts['learning_rate'],
                             weight_decay=1e-5)

acc = test_model(classifier, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X
    if opts['add_noise']: inputs += torch.randn(inputs.shape)/200
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
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    # ===================log========================
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.data[0]))
  if epoch % 1 == 0:
    autoencoder_model.eval()
    acc = test_model_on_gen(classifier, autoencoder_model, test_loader)
    autoencoder_model.train()
    #img_vis = train_dataset[0][0:bs]
    #img = Variable(img_vis).cuda()
    #output = autoencoder_model(img)
    #out = torch.cat((img, output), 0)
    #pic = to_img(out.cpu().data)
    #save_image(pic, './temp_images/image_{}.png'.format(epoch))
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
