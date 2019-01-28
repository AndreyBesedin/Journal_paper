import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

opts = {
  'batch_size': 100,
  'learning_rate': 0.001
  }

class Classifier_MNIST_512_features(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_MNIST_512_features, self).__init__()
    self.fc1 = nn.Linear(512, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    #x = F.softmax(self.fc3(x))
    return x
  
class autoencoder_MNIST_512_features(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(autoencoder_MNIST_512_features, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(512, 128),
      linear_block(128, 64),
      nn.Linear(64, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 64),
      linear_block(64, 128),
      nn.Linear(128, 512),
      nn.Tanh()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
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

def test_classifier_on_generator(classif, gen_model, data_loader):
  #TODO add cases of models other than AE
  total = 0
  correct = 0
  gen_model.eval()
  for idx, (test_X,  test_Y) in enumerate(data_loader):
    input_test = gen_model(test_X.float().cuda())
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  gen_model.train()
  return correct/total*100

def get_indices_for_classes(data, data_classes):
  # Creates a list of indices of samples from the dataset, corresponding to given classes
  indices = torch.FloatTensor(list((data.tensors[1].long()==class_).tolist() for class_ in data_classes)).sum(0).nonzero().long().squeeze()
  return indices[torch.randperm(len(indices))]

# Loading the datasets
full_trainset = torch.load('./data/trainset.pth')
full_testset = torch.load('./data/testset.pth')

trainset = TensorDataset(full_trainset[0], full_trainset[1])
testset = TensorDataset(full_testset[0], full_testset[1])

# Initializing data loaders for first 5 classes
pretrain_on_classes = range(5)

indices_train = get_indices_for_classes(trainset, pretrain_on_classes)
indices_test = get_indices_for_classes(testset, pretrain_on_classes)
train_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_train))
test_loader = DataLoader(testset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_test))

# Initializing classification model
classifier = Classifier_MNIST_512_features(10)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.5, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()
classifier.cuda()
classification_criterion.cuda()

gen_model = autoencoder_MNIST_512_features(32)
gen_dict = torch.load('./pretrained_models/AE_5_classes_32_code_size.pth')
gen_model.load_state_dict(gen_dict)
gen_model.cuda()

acc = test_classifier(classifier, test_loader)
print('Classification accuracy prior to training: {:.4f}'.format(acc))

max_accuracy = 0
for epoch in range(10):
  for idx, (X, Y) in enumerate(train_loader):
    inputs = gen_model(X.cuda())
    labels = Y.long().cuda()
    outputs = classifier(inputs.data)
    classification_loss = classification_criterion(outputs, labels)
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    if idx%50==0:
      print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, 10,  classification_loss.item()))
  acc = test_classifier(classifier, test_loader)
  print('Test accuracy after {} epochs: {:.8f}'.format(epoch+1, acc))    
  if acc > max_accuracy:
    max_accuracy = acc
    torch.save(classifier.state_dict(), './pretrained_models/classifier_5_classes_reconstructed_data.pth')
      
    
    