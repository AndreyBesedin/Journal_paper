import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

opts = {
  'batch_size': 1000,
  'learning_rate': 0.001
  }

class Data_Buffer:
  def __init__(self, max_batches_per_class=60, batch_size=100):
    self.max_batches_per_class = max_batches_per_class
    self.batch_size = batch_size
    self.dbuffer = {}
    self.oldest_batches = {}
  
  def add_batch(self, batch, class_label):
    """
    Adding a batch to the buffer to the corresponding class storage, 
    The oldest batches are replaced when the buffer is full
    If an unknown class appears - a new storage is added
    """
    if str(class_label) in self.dbuffer.keys():
      if len(self.dbuffer[str(class_label)]) < self.max_batches_per_class:
        #print('adding new batch')
        self.dbuffer[str(class_label)].append(batch.clone())
      else:
        self.dbuffer[str(class_label)][self.oldest_batches[str(class_label)]] = batch.clone()
        self.oldest_batches[str(class_label)] = (self.oldest_batches[str(class_label)] + 1) % self.max_batches_per_class
        #print('replacing old batch')
    else:
      #print('Class label:{}'.format(class_label))
      #print('initializing new class')
      self.dbuffer[str(class_label)] = [batch.clone()]
      self.oldest_batches[str(class_label)] = 0
      
  def transform_data(self, transform):
    # Inplace apply a given transform to all the batches in the buffer
    for class_label in self.dbuffer.keys():
      for idx in range(len(self.dbuffer[str(class_label)])):
        self.dbuffer[str(class_label)][idx] = transform(self.dbuffer[str(class_label)][idx]).data
        
  def make_tensor_dataset(self):
    # Transform the buffer into a single tensor dataset
    tensor_data = []
    tensor_labels = []
    for key in self.dbuffer.keys():
      tensor_data += self.dbuffer[key]
      tensor_labels += [int(key)]*(self.batch_size*len(self.dbuffer[key]))
    tensor_data = torch.stack(tensor_data)
    tensor_data = tensor_data.reshape(tensor_data.shape[0]*tensor_data.shape[1], tensor_data.shape[2])
    return TensorDataset(tensor_data, torch.FloatTensor(tensor_labels))
  
class Classifier_2048_features(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_2048_features, self).__init__()
    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
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
  return indices[torch.randperm(len(indices))]

# Loading the datasets
full_trainset = torch.load('./data/trainset_no_relu.pth')
full_testset = torch.load('./data/testset_no_relu.pth')

trainset = TensorDataset(full_trainset[0], full_trainset[1])
testset = TensorDataset(full_testset[0], full_testset[1])

# Initializing data loaders for first 5 classes
pretrain_on_classes = range(15)

indices_train = get_indices_for_classes(trainset, pretrain_on_classes)
indices_test = get_indices_for_classes(testset, pretrain_on_classes)
train_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_train))
test_loader = DataLoader(testset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_test))

# Initializing classification model, criterion and optimizer function
classifier = Classifier_2048_features(30)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()
classifier.cuda()
classification_criterion.cuda()

max_accuracy = 0
training_epochs = 10
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
    torch.save(classifier.state_dict(), './pretrained_models/classifier_15_classes_original_data.pth')
    
    
