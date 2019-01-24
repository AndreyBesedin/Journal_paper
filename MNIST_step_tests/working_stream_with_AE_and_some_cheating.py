import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix

opts = {
  'batch_size': 100,
  'learning_rate': 0.001,
  'betta1': 1e-2, # Importance coefficient for classification loss in AE
  'betta2': 1, # Importance coefficient for reconstruction loss in AE
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
    outputs = classif(input_test)
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
  return indices[torch.randperm(len(indices))]

# Loading the datasets
full_trainset = torch.load('./data/trainset.pth')
full_testset = torch.load('./data/testset.pth')

trainset = TensorDataset(full_trainset[0], full_trainset[1])
testset = TensorDataset(full_testset[0], full_testset[1])

# Initializing data loaders for first 5 classes
pretrained_on_classes = range(5)
new_class = [5]

indices_fake = get_indices_for_classes(trainset, pretrained_on_classes)
indices_real = get_indices_for_classes(trainset, new_class)
indices_test = get_indices_for_classes(testset, range(6))
fake_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_fake))
real_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_real), drop_last=True)
test_loader = DataLoader(testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))

# Initializing classification model
classifier = Classifier_MNIST_512_features(10)
class_dict = torch.load('./pretrained_models/classifier_5_classes_reconstructed_data.pth')
classifier.load_state_dict(class_dict)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.5, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()
classifier.cuda()
classification_criterion.cuda()

gen_model = autoencoder_MNIST_512_features(32)
gen_dict = torch.load('./pretrained_models/AE_5_classes_32_code_size.pth')
gen_model.load_state_dict(gen_dict)
gen_model.cuda()

generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.5, 0.999), weight_decay=1e-5)
generative_criterion_cl = nn.MSELoss()
generative_criterion_cl.cuda()
generative_criterion_rec = nn.MSELoss()
generative_criterion_rec.cuda()

acc = test_classifier(classifier, test_loader)
print('Classification accuracy prior to training: {:.4f}'.format(acc))

max_accuracy = 0
fake_batches = 20
real_batches = 1
known_classes = [0, 1, 2, 3, 4]

for epoch in range(100):
  stream_class = [np.random.randint(0,10)]
  indices_fake = get_indices_for_classes(trainset, known_classes)
  indices_real = get_indices_for_classes(trainset, stream_class)
  if stream_class[0] not in known_classes:
    print('ADDED A NEW CLASS {}'.format(stream_class))
    known_classes.append(stream_class[0])
    indices_test = get_indices_for_classes(testset, known_classes)
    test_loader = DataLoader(testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))
    acc_real = test_classifier(classifier, test_loader)
    acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
    print('Real test accuracy after {} epochs with a new class: {:.8f}'.format(epoch+1, acc_real))    
    print('Reconstructed test accuracy after {} epochs with a new class: {:.8f}'.format(epoch+1, acc_fake))     
  fake_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_fake))
  real_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_real), drop_last=True)
  
  for idx_real, (X_real, Y_real) in enumerate(real_loader):
    mixed_batch_data = []
    mixed_batch_labels = []
    mixed_batch_data.append(X_real.cuda())
    mixed_batch_labels.append(Y_real.long().cuda())
    for idx_fake, (X_fake, Y_fake) in enumerate(fake_loader):
      if idx_fake>=fake_batches:
        mixed_batch_data.append(X_fake.cuda())
        mixed_batch_labels.append(Y_fake.long().cuda())
        break
      mixed_batch_data.append(gen_model(X_fake.cuda()))
      mixed_batch_labels.append(Y_fake.long().cuda())
    nb_of_batches = len(mixed_batch_labels)
    inputs = torch.stack(mixed_batch_data).reshape(opts['batch_size']*(nb_of_batches), 512)
    labels = torch.stack(mixed_batch_labels).reshape(opts['batch_size']*(nb_of_batches))
    
    # Updating the classifier
    outputs = classifier(inputs.data)
    classification_loss = classification_criterion(outputs, labels)
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    if idx_real%100==0:
      print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, 10,  classification_loss.item()))
    # Updating the auto-encoder
    
    outputs = gen_model(inputs.data)
    orig_classes = classifier(inputs.data)
    classification_reconstructed = classifier(outputs)
    loss_gen_rec = opts['betta2']*generative_criterion_rec(outputs, inputs.data)
    loss_gen_cl = opts['betta1']*generative_criterion_cl(classification_reconstructed, orig_classes.data)
    #print('reconstruction loss: {}'.format(loss_gen_rec.item()))
    #print('classification loss: {}'.format(loss_gen_cl.item()))
    loss_gen = loss_gen_cl + loss_gen_rec
    loss_gen.backward()
    generative_optimizer.step()
    generative_optimizer.zero_grad()    

  acc_real = test_classifier(classifier, test_loader)
  acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Real test accuracy after {} epochs: {:.8f}'.format(epoch+1, acc_real))    
  print('Reconstructed test accuracy after {} epochs: {:.8f}'.format(epoch+1, acc_fake))    
  #if acc > max_accuracy:
    #max_accuracy = acc
    #torch.save(classifier.state_dict(), './pretrained_models/classifier_6_classes_mixed_data.pth')
      
    
    
