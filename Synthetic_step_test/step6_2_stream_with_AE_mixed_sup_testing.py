
"""
Working version of data stream classification training on LSUN 
"""
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix

from data_buffer import Data_Buffer

opts = {
  'batch_size': 50,
  'learning_rate': 0.001,
  'betta1': 1e-2, # Influence coefficient for classification loss in AE default 1e-2
  'betta2': 1, # Influence coefficient for reconstruction loss in AE
  }

torch.cuda.set_device(3)
classes_per_interval = 20
nb_of_classes = 500
feature_size = 128
code_size = 32
fake_batches = 20
real_batches = 1
real_buffer_size = 1
real_buffer = Data_Buffer(real_buffer_size, opts['batch_size'])
name_to_save = './results/res_stream_{}_fake_batches_{}_hist_batches_{}_batches_in_storage_{}_betta1_{}_betta2.pth'.format(fake_batches, real_batches, real_buffer_size, opts['betta1'], opts['betta2'])

class Classifier_128_features(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_128_features, self).__init__()
    self.fc1 = nn.Linear(128, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x  
  
class autoencoder_128_features(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(autoencoder_128_features, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      linear_block(128, 64),
      nn.Linear(64, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 64),
      linear_block(64, 128),
      nn.Linear(128, 128),
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
    outputs = classif(input_test.data)
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

# ----------------------------------------- LOADING THE ORIGINAL DATASETS ------------------------------------------------------
full_data = torch.load('./data/data_train_test_500_classes_128_features_2000_samples.pth')
trainset = TensorDataset(full_data['data_train'], full_data['labels_train'])
testset = TensorDataset(full_data['data_test'], full_data['labels_test'])

# Initializing data loaders for first 5 classes


# ----------------------------------------- LOADING PRETRAINED MODELS ----------------------------------------------------------
# Initializing classification model
full_classifier = Classifier_128_features(nb_of_classes)
full_class_dict = torch.load('./pretrained_models/full_classifier_500_classes_original_data.pth')
full_classifier.load_state_dict(full_class_dict)

classifier = Classifier_128_features(nb_of_classes)
class_dict = torch.load('./pretrained_models/classifier_250_classes_reconstructed_data.pth')
classifier.load_state_dict(class_dict)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()
classifier.cuda()
classification_criterion.cuda()

gen_model = autoencoder_128_features(code_size)
gen_dict = torch.load('./pretrained_models/AE_250_classes_32_code_size.pth')
gen_model.load_state_dict(gen_dict)
gen_model.cuda()

generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
generative_criterion_cl = nn.MSELoss()
generative_criterion_cl.cuda()
generative_criterion_rec = nn.MSELoss()
generative_criterion_rec.cuda()

# ---------------------------------- FILLING THE BUFFERS WITH THE HISTORICAL DATA ----------------------------------------------
prev_classes = list(range(250))
historical_buffer = Data_Buffer(60, opts['batch_size'])
for idx_class in prev_classes:
  indices_prev = get_indices_for_classes(trainset, [idx_class])
  prev_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_prev),  drop_last=True)
  for batch, label in prev_loader:                                                                                
    historical_buffer.add_batch(gen_model.encoder(batch.cuda()).data, idx_class)
    real_buffer.add_batch(batch.cuda(), idx_class)

max_accuracy = 0

known_classes = [int(a) for a in historical_buffer.dbuffer.keys()]
indices_test = get_indices_for_classes(testset, known_classes)
test_loader = DataLoader(testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))

acc_real = test_classifier(classifier, test_loader)
acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
results = {}
results['accuracies'] = []
results['known_classes'] = []
print('Real test accuracy on known classes prior to stream training: {:.8f}'.format(acc_real))    
print('Reconstructed test accuracy on known classes prior to stream training: {:.8f}'.format(acc_fake))  
results['accuracies'].append(acc_real)
results['known_classes'].append(len(known_classes))

# --------------------------------------------------- STREAM TRAINING ----------------------------------------------------------

stream_duration = 1000
for interval in range(stream_duration):
  gen_model_old = copy.deepcopy(gen_model) 
  stream_classes = [np.random.randint(0, 500) for idx in range(classes_per_interval)]
  stream_indices = get_indices_for_classes(trainset, stream_classes)
    
  # Preloading the historical data/stored codes 
  fake_data = historical_buffer.make_tensor_dataset()
  real_data = real_buffer.make_tensor_dataset()
  
  stream_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(stream_indices), drop_last=True)
  fake_loader = DataLoader(fake_data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
  real_loader = DataLoader(real_data, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
  known_classes = [int(a) for a in historical_buffer.dbuffer.keys()]
  
  # We will make an extra testing step with completed testset if a new class is coming from the stream
  added_new_classes = 0
  for stream_class in stream_classes:
    if stream_class not in known_classes:
      added_new_classes += 1
      print('ADDED A NEW CLASS {}'.format(stream_class))
      known_classes.append(stream_class)
  if added_new_classes>0:
    print('Added {} new classes, reevaluating the classifiers performance'.format(added_new_classes))
    print('Currently known {} classes'.format(len(known_classes)))
    indices_test = get_indices_for_classes(testset, known_classes)
    test_loader = DataLoader(testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))
    acc_real = test_classifier(classifier, test_loader)
    acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
    print('Real test accuracy with new classes: {:.8f}'.format(acc_real))    
    print('Reconstructed test accuracy with new classes: {:.8f}'.format(acc_fake))    
    results['accuracies'].append(acc_real)
    results['known_classes'].append(len(known_classes))

  
  for idx_stream, (X_stream, Y_stream) in enumerate(stream_loader):
    mixed_batch_data = []
    mixed_batch_labels = []
    mixed_batch_data.append(X_stream.cuda())
    mixed_batch_labels.append(Y_stream.long().cuda())
    for idx_fake, (X_fake, Y_fake) in enumerate(fake_loader):
      if idx_fake>=fake_batches-1:
        break
      mixed_batch_data.append(gen_model_old.decoder(X_fake.cuda()).data)
      mixed_batch_labels.append(Y_fake.long().cuda())
    for idx_real, (X_real, Y_real) in enumerate(real_loader):
      if idx_real>=real_batches-1:
        break
      mixed_batch_data.append(X_real.cuda())
      mixed_batch_labels.append(Y_real.long().cuda())
      
    nb_of_batches = len(mixed_batch_labels)
    #print('Batches forming a big one: {}'.format(nb_of_batches))
    inputs = torch.stack(mixed_batch_data).reshape(opts['batch_size']*nb_of_batches, feature_size)
    labels = torch.stack(mixed_batch_labels).reshape(opts['batch_size']*nb_of_batches)
    micro_testset = TensorDataset(inputs, labels)
    micro_test_loader = DataLoader(micro_testset, batch_size=100, shuffle=False)
    
    # Updating the classifier
    outputs = classifier(inputs)
    classification_loss = classification_criterion(outputs, labels)
    #classification_loss.backward(retain_graph=True)
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    
    # Updating the auto-encoder
    for idx in range(10):
      train_acc = test_classifier_on_generator(full_classifier, gen_model, micro_test_loader)
      print('Training accuracy on reconstructed data: {}'.format(train_acc))
      reconstructions = gen_model(inputs)
      orig_classes = classifier(inputs).detach()
      classification_reconstructed = classifier(reconstructions)
      loss_gen_rec = opts['betta2']*generative_criterion_rec(reconstructions, inputs)
      loss_gen_cl = opts['betta1']*generative_criterion_cl(classification_reconstructed, orig_classes)
    #print('reconstruction loss: {}'.format(loss_gen_rec.item()))
    #print('classification loss: {}'.format(loss_gen_cl.item()))
      loss_gen = loss_gen_cl + loss_gen_rec
      loss_gen.backward()
      generative_optimizer.step()
      generative_optimizer.zero_grad()   

    
    #if idx_stream%100==0:
      #print('interval [{}/{}], classification loss: {:.4f}'.format(interval, stream_duration,  classification_loss.item()))
      #acc_real = test_classifier(classifier, test_loader)
      #acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
      #results['accuracies'].append(acc_real)
      #results['known_classes'].append(len(known_classes))
      #torch.save(results, name_to_save)
      
  # At the end of each interval we update the code storage  
  historical_buffer.transform_data(nn.Sequential(gen_model_old.decoder, gen_model.encoder))
  # And store the encoded streaming data + the latest streaming batch in original shape
  for stream_class in stream_classes:
    class_indices = get_indices_for_classes(trainset, [stream_class])
    class_loader =  DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(class_indices), drop_last=True)
    for X_real, Y_real in class_loader:
      historical_buffer.add_batch(gen_model.encoder(X_real.cuda()).data, stream_class)
      real_buffer.add_batch(X_real.cuda(), stream_class)

  acc_real = test_classifier(classifier, test_loader)
  acc_fake = test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Real test accuracy after {} intervals: {:.8f}'.format(interval+1, acc_real))    
  print('Reconstructed test accuracy after {} intervals: {:.8f}'.format(interval+1, acc_fake))   
  results['accuracies'].append(acc_real)
  results['known_classes'].append(len(known_classes))
  torch.save(results, name_to_save)
  
  #if acc > max_accuracy:
    #max_accuracy = acc
    #torch.save(classifier.state_dict(), './pretrained_models/classifier_6_classes_mixed_data.pth')

