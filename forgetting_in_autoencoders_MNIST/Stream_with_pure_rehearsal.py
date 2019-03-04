
"""
Working version of data stream classification training on LSUN 
"""
import os
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
import models
import sup_functions

opts = {
  'batch_size': 60,
  'learning_rate': 0.001,
  'betta1': 0.1,
  'betta2': 1,
  }

cuda_device = 1
torch.cuda.set_device(cuda_device)
nb_of_classes = 10
classes_per_interval = 1
feature_size = 28*28
max_interval_duration = 100 # Maximum number of images in each stream interval, then change the environment
real_batches = 5
fake_batches = 0
real_buffer_size = 1 # (* batch size = number of real data samples we store for further training)
name_to_save = './results/MNIST_stream_{}_fake_batches_{}_hist_batches_{}_batches_in_storage_{}_betta1_{}_betta2.pth'.format(fake_batches, real_batches, real_buffer_size, opts['betta1'], opts['betta2'])

# ----------------------------------------- LOADING THE ORIGINAL DATASETS ------------------------------------------------------

trainset = torch.load('./data/trainset.pth')
testset = torch.load('./data/testset.pth')
full_original_trainset = TensorDataset(trainset[0], trainset[1], torch.zeros(len(trainset[1])))
full_original_testset = TensorDataset(testset[0], testset[1])
del trainset, testset


# ----------------------------------------- LOADING PRETRAINED MODELS ----------------------------------------------------------
# Initializing classification model
classifier = models.Classifier_MNIST(nb_of_classes)
class_dict = torch.load('./pretrained_models/classifier_5_MNIST.pth')
classifier.load_state_dict(class_dict)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()
classifier.cuda()
classification_criterion.cuda()

# ---------------------------------- FILLING THE BUFFERS WITH THE HISTORICAL DATA ----------------------------------------------

real_buffer = Data_Buffer(real_buffer_size, opts['batch_size'])
real_buffer.cuda_device = cuda_device

"""
Filling in historical buffers
"""
pretrained_on_classes = list(range(5))
real_buffer.load_from_tensor_dataset(full_original_trainset, pretrained_on_classes)

print('Historical data successfully loaded from the datasets')

max_accuracy = 0
print('Testing already acquired knowledge prior to stream training')

known_classes = pretrained_on_classes 
indices_test = sup_functions.get_indices_for_classes(full_original_testset, known_classes)
test_loader = DataLoader(full_original_testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))

acc_real = sup_functions.test_classifier(classifier, test_loader)

results = {}
results['accuracies'] = []
results['known_classes'] = []
print('Real test accuracy on known classes prior to stream training: {:.8f}'.format(acc_real))    
results['accuracies'].append(acc_real)
results['known_classes'].append(len(known_classes))

# --------------------------------------------------- STREAM TRAINING ----------------------------------------------------------


stream_duration = 1000
for interval in range(stream_duration):
  #gen_model_old = copy.deepcopy(gen_model) 
  stream_classes = [np.random.randint(0, nb_of_classes) for idx in range(classes_per_interval)]
  #stream_indices = sup_functions.get_indices_for_classes(full_original_trainset, stream_classes)
  
  # Preloading the historical data/stored codes
  print('Starting new interval, create tensors from storages') 
  #sup_functions.memory_check()
  #codes_storage.make_tensor_dataset()
  real_buffer.make_tensor_dataset()
  print('Fake and real tensors created')
  #sup_functions.memory_check()
  # Initializing loaders: stream, historical and codes to decode
  stream_loaders = {}
  #stream_loader = DataLoader(full_original_trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(stream_indices), drop_last=True)
  #fake_loader = DataLoader(codes_storage.tensor_dataset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
  real_loader = DataLoader(real_buffer.tensor_dataset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
  known_classes = [int(a) for a in real_buffer.dbuffer.keys()]
  
  # We will make an extra testing step with completed testset if a new class is coming from the stream
  added_new_classes = 0
  for stream_class in stream_classes:
    stream_indices = sup_functions.get_indices_for_classes(full_original_trainset, [stream_class])
    stream_loaders[str(stream_class)] = DataLoader(full_original_trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(stream_indices), drop_last=True)
    if stream_class not in known_classes:
      added_new_classes += 1
      print('ADDED A NEW CLASS {}'.format(stream_class))
      known_classes.append(stream_class)
  if added_new_classes>0:
    print('Added {} new classes, reevaluating the classifiers performance'.format(added_new_classes))
    print('Currently known {} classes'.format(len(known_classes)))
    indices_test = sup_functions.get_indices_for_classes(full_original_testset, known_classes)
    test_loader = DataLoader(full_original_testset, batch_size=1000, sampler = SubsetRandomSampler(indices_test))
    acc_real = sup_functions.test_classifier(classifier, test_loader)
    #acc_fake = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
    print('Real test accuracy with new classes: {:.8f}'.format(acc_real))    
    #print('Reconstructed test accuracy with new classes: {:.8f}'.format(acc_fake))    
    results['accuracies'].append(acc_real)
    results['known_classes'].append(len(known_classes))

  #for idx_stream, (X_stream, Y_stream, _) in enumerate(stream_loader):
  for idx_stream in range(max_interval_duration):
    # Forming a mixed data batch
    #if idx_stream*opts['batch_size'] > max_interval_duration:
    #  break
    current_class = stream_classes[np.random.randint(classes_per_interval)]
    (X_stream, Y_stream, _) = next(iter(stream_loaders[str(current_class)]))
    mixed_batch_data = []
    mixed_batch_labels = []
    mixed_batch_reconstruction_counter = []
    mixed_batch_data.append(X_stream.cuda())
    mixed_batch_labels.append(Y_stream.long().cuda())
    for idx_real in range(real_batches):
      real_batch = next(iter(real_loader))
      mixed_batch_data.append(real_batch[0].cuda())
      mixed_batch_labels.append(real_batch[1].long().cuda())
    real_buffer.add_batch(X_stream, current_class, 0)   
    nb_of_batches = len(mixed_batch_labels)
    #print('Batches forming a big one: {}'.format(nb_of_batches))
    inputs = torch.stack(mixed_batch_data).reshape(opts['batch_size']*nb_of_batches, feature_size)
    labels = torch.stack(mixed_batch_labels).reshape(opts['batch_size']*nb_of_batches)
  
    # Updating the classifier
    outputs = classifier(inputs)
    #classification_loss = classification_criterion(outputs, labels)
    classification_loss = classification_criterion(outputs, labels)
    #classification_loss.backward(retain_graph=True)
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
#    if idx_stream%20==0:
#      print('interval [{}/{}], classification loss: {:.4f}'.format(interval, stream_duration,  classification_loss.item()))
      #acc_real = sup_functions.test_classifier(classifier, test_loader)
      #acc_fake = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
      #results['accuracies'].append(acc_real)
      #results['known_classes'].append(len(known_classes))
      #torch.save(results, name_to_save)
    # Updating the auto-encoder
    
#    reconstructions = gen_model(inputs)
#    orig_classes = classifier(inputs).detach()
#    classification_reconstructed = classifier(reconstructions)
#    loss_gen_rec = opts['betta2']*sup_functions.MSEloss_weighted(reconstructions, inputs, reconstruction_counter)
#    loss_gen_cl = opts['betta1']*sup_functions.MSEloss_weighted(classification_reconstructed, orig_classes, reconstruction_counter)
#    loss_gen_rec = opts['betta2']*generative_criterion_rec(reconstructions, inputs)
#    loss_gen_cl = opts['betta1']*generative_criterion_cl(classification_reconstructed, orig_classes)
    #print('reconstruction loss: {}'.format(loss_gen_rec.item()))
    #print('classification loss: {}'.format(loss_gen_cl.item()))
#    loss_gen = loss_gen_cl + loss_gen_rec
#    loss_gen.backward()
#    generative_optimizer.step()
#    generative_optimizer.zero_grad()   
    if idx_stream%20==0:
      print('interval [{}/{}], classification loss: {:.4f}'.format(interval, stream_duration,  classification_loss.item()))
          
  # At the end of each interval we update the code storage  
#  codes_storage.transform_data(nn.Sequential(gen_model_old.decoder, gen_model.encoder))
  # And store the encoded streaming data + the latest streaming batch in original shape
  #real_buffer.load_from_tensor_dataset(full_original_trainset, stream_classes)
  #codes_storage.load_from_tensor_dataset(full_original_trainset, stream_classes, gen_model.encoder)

  acc_real = sup_functions.test_classifier(classifier, test_loader)
  #acc_fake = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Real test accuracy after {} intervals: {:.8f}'.format(interval+1, acc_real))    
  #print('Reconstructed test accuracy after {} intervals: {:.8f}'.format(interval+1, acc_fake))   
  results['accuracies'].append(acc_real)
  results['known_classes'].append(len(known_classes))
  torch.save(results, name_to_save)
  
  #if acc > max_accuracy:
    #max_accuracy = acc
    #torch.save(classifier.state_dict(), './pretrained_models/classifier_6_classes_mixed_data.pth')

