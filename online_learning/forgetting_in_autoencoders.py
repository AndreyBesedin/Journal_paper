import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset
from progress.bar import Bar

cuda_device_number = 0
torch.cuda.set_device(cuda_device_number)
ls = 32

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


class autoencoder(nn.Module):
  def __init__(self):
    global ls
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 32, 5, 1, 2),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2), 
      nn.Conv2d(32, 64, 5, 1, 2),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
      nn.Conv2d(64, ls, 5, 1, 2),
      nn.BatchNorm2d(ls),
      nn.ReLU(True),
      nn.MaxPool2d(7, stride=None)  # b, 8, 2, 2
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(ls, 32, 7, stride=1, padding=0),  # b, 16, 5, 5
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 64, 6, stride=2, padding=1),  # b, 8, 15, 15
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 16, 7, stride=1, padding=0),  # b, 1, 28, 28
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 1, 7, stride=1, padding=0),  # b, 1, 28, 28
      nn.Tanh()
    )
  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 28,28)
    x = self.encoder(x)
    x = self.decoder(x)
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

def reconstruct_full_dataset_with_AE(train_dataset, model):
  bs=1000
  data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
  res_data = torch.zeros(train_dataset.train_data.shape).reshape(60000,1, 28, 28)
  res_labels = torch.zeros(train_dataset.train_labels.shape[0])
  current_index = 0
  bar = Bar('Reconstructing data for absent classes:', max=60)
  for idx, (train_x, train_y) in enumerate(data_loader):
      #call('nvidia-smi')
    bar.next()
    inputs = train_x.cuda()
    batch = model(inputs)
    current_batch_size = batch.shape[0]
    res_data[current_index:current_index+current_batch_size] = batch.cpu().data
    res_labels[current_index:current_index+current_batch_size] = train_y
    current_index+=current_batch_size 
  bar.finish()
  return (res_data, res_labels)

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

def test_model_on_reconstructed(model, autoencoder, test_loader):
  total = 0
  correct = 0
  conf_matrix = np.zeros((10,10))
  for idx, (test_x, test_y) in enumerate(test_loader):
    input_test = autoencoder(test_x.cuda())
    outputs = model(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
    conf_matrix+=confusion_matrix(predicted.cpu().numpy(), labels.numpy())
  return correct/total*100, (conf_matrix/conf_matrix.sum(0)*100).transpose(0,1)

def main():
  classifier = Net().cuda()
  generative_model = autoencoder().cuda()
  
  criterion_classifier = nn.CrossEntropyLoss().cuda()
  criterion_G_classif = nn.MSELoss()
  criterion_G_standard = nn.MSELoss()
  
  optimizer_classifier = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.99)
  optimizer_G_standard = torch.optim.Adam(generative_model.parameters(), lr=0.001,
                             weight_decay=1e-5)
  optimizer_G_classif = torch.optim.Adam(generative_model.parameters(), lr=0.001*0.2,
                             weight_decay=1e-5)
  max_test_acc = 0

  img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  progress_to_save = {
          'confusion': [],
          'classes': []
          }
  root = '~/workspace/Projects/Journal_paper/'
  original_dataset = datasets.MNIST(root=root+'datasets/MNIST/',
    train=True,
    download=True,
    transform=img_transform)
  test_dataset = datasets.MNIST(root=root+'datasets/MNIST/',
    train=False,
    download=True,
    transform=img_transform)

  test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
  test_acc = test_model(classifier, test_loader)

  #train_dataset_ = torch.load('../datasets/LSUN_features/testset.pt')
  #train_dataset = TensorDataset(train_dataset_[0], train_dataset_[1])
  train_loader = DataLoader(original_dataset, batch_size=100, shuffle=True)
  train_epochs = 25
  for train_epoch in range(train_epochs):
    running_loss = 0.0
    bar = Bar('Training epoch '+str(train_epoch)+':', max=int(600))
    for idx, (train_x, train_y) in enumerate(train_loader):
      bar.next()
      inputs = train_x.cuda()
      labels = train_y.cuda()
      # zero the parameter gradients
      optimizer_classifier.zero_grad()
      # forward + backward + optimize
      outputs = classifier(inputs)
      loss = criterion_classifier(outputs, labels.long())
      loss.backward()
      optimizer_classifier.step()
      outputs = outputs.detach()
      # Optimizing generative model
      #img = Variable(train_x).cuda()
      rec_data = generative_model(inputs)
      classification_reconstructed = classifier(rec_data)
      
      loss_classif = criterion_G_classif(classification_reconstructed, outputs)
      optimizer_G_classif.zero_grad()
      loss_classif.backward(retain_graph=True)
      optimizer_G_classif.step()
      
      loss_standard = criterion_G_standard(rec_data, inputs)
      optimizer_G_standard.zero_grad()
      loss_standard.backward()
      optimizer_G_standard.step()
        # print statistics
      running_loss += loss.item()
      if idx % 100 == 99:    # print every 2000 mini-batches
        #print('[%d, %5d] loss: %.3f' %
              #(retrain_epoch + 1, retrain_epoch + 1, running_loss / 100))
        running_loss = 0.0
        
    test_acc, confusion = test_model(classifier, test_loader)
    test_acc_rec, confusion_rec = test_model_on_reconstructed(classifier, generative_model, test_loader)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = classifier.float()
      #torch.save(best_model, 'models/'+ dataset +'_classifier.pt')
    bar.finish()
    progress_to_save['confusion'].append(confusion_rec)
    progress_to_save['classes'].append(-1)
    #print('Test on original testset')
    #print(confusion)
    #print('Test on reconstructed testset')
    #print(confusion_rec)
    torch.save(progress_to_save, './results/train_progress_forgetting_ae_'+str(ls)+'.pth')
    #if retrain_epoch%20==0:
      #torch.save(generative_model.state_dict(), './results/models/progress_' + str(retrain_epoch) + '.pth')

    print('Test accuracy: ' + str(test_acc) + '; Reconstructed accuracy: ' + str(test_acc_rec))
  
  retrain_epochs=500
  for retrain_epoch in range(retrain_epochs):
    reconstructed_dataset = reconstruct_full_dataset_with_AE(original_dataset, generative_model)
    train_dataset = TensorDataset(reconstructed_dataset[0], reconstructed_dataset[1])
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    running_loss = 0.0
    bar = Bar('Retraining epoch '+str(retrain_epoch)+':', max=int(600))
    for idx, (train_x, train_y) in enumerate(train_loader):
      bar.next()
      inputs = train_x.cuda()
      labels = train_y.cuda()
      # zero the parameter gradients
      optimizer_classifier.zero_grad()
      # forward + backward + optimize
      outputs = classifier(inputs)
      loss = criterion_classifier(outputs, labels.long())
      loss.backward()
      optimizer_classifier.step()
      outputs = outputs.detach()
      # Optimizing generative model
      #img = Variable(train_x).cuda()
      rec_data = generative_model(inputs)
      classification_reconstructed = classifier(rec_data)
      
      loss_classif = criterion_G_classif(classification_reconstructed, outputs)
      optimizer_G_classif.zero_grad()
      loss_classif.backward(retain_graph=True)
      optimizer_G_classif.step()
      
      loss_standard = criterion_G_standard(rec_data, inputs)
      optimizer_G_standard.zero_grad()
      loss_standard.backward()
      optimizer_G_standard.step()
        # print statistics
      running_loss += loss.item()
      if idx % 100 == 99:    # print every 2000 mini-batches
        #print('[%d, %5d] loss: %.3f' %
              #(retrain_epoch + 1, retrain_epoch + 1, running_loss / 100))
        running_loss = 0.0
        
    test_acc, confusion = test_model(classifier, test_loader)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = classifier.float()
      #torch.save(best_model, 'models/'+ dataset +'_classifier.pt')
    bar.finish()
    progress_to_save['confusion'].append(confusion)
    progress_to_save['classes'].append(1)
    print(confusion)
    torch.save(progress_to_save, './results/train_progress_forgetting_ae_'+str(ls)+'.pth')
    #if retrain_epoch%20==0:
      #torch.save(generative_model.state_dict(), './results/models/progress_' + str(retrain_epoch) + '.pth')

    print('Test accuracy: ' + str(test_acc))

if __name__ == '__main__':
  main()
