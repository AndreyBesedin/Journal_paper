import time
import copy
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class Data_Buffer:
  def __init__(self, max_batches_per_class=60, batch_size=100):
    self.max_batches_per_class = max_batches_per_class
    self.batch_size = batch_size
    self.dbuffer = {}
    self.oldest_batches = {}
    self.cuda_device = 0
    self.data_stats = []
    self.mean_drift = []
    self.compute_stats = False
  
  def compute_data_drift(self):
    if len(self.data_stats)<=1:
      print('Not enough history to compute drift, come back later=)')
    else:
      drift_from_last = {key: (self.data_stats[-2]['by_class'][key]['mean'] - self.data_stats[-1]['by_class'][key]['mean']).norm().item() for key in self.data_stats[-1]['by_class'].keys()}
      drift_from_orig = {key: (self.data_stats[0]['by_class'][key]['mean'] - self.data_stats[-1]['by_class'][key]['mean']).norm().item() for key in self.data_stats[-1]['by_class'].keys()}
      total_drift_from_last = (self.data_stats[-2]['full_mean'] - self.data_stats[-1]['full_mean']).norm().item()
      total_drift_from_orig = (self.data_stats[0]['full_mean'] - self.data_stats[-1]['full_mean']).norm().item()
      print('Classwise mean drift from last epoch:')
      print(sum(list(drift_from_last.values()))/len(drift_from_last.values()))
      print('Classwise mean drift from the original distribution:') 
      print(sum(list(drift_from_orig.values()))/len(drift_from_orig.values()))
      print('Total mean drift from last epoch:')
      print(total_drift_from_last)
      print('Total mean drift from the original distribution:')
      print(total_drift_from_orig)
      self.mean_drift.append({
        'drift_from_last': drift_from_last,
        'total_drift_from_last': total_drift_from_last, 
        'drift_from_orig': drift_from_orig,
        'total_drift_from_orig': total_drift_from_orig
      })

  def compute_data_stats(self):
    """
    Tool to compute current data basic statistics
    For each data class we compute the center of the cloud of corresponding points and its coordinate-wise standard deviation ('by class' key)
    We also compute the center of the whole samples distribution in the dataset ('full_mean' key) 
    """
    latest_stats = {}
    total_mean = 0
    for key in self.dbuffer.keys():
      latest_stats[key] = {
        'mean': (sum(self.dbuffer[key]['data'])/len(self.dbuffer[key]['data'])).mean(0),
        'std' : (sum(self.dbuffer[key]['data'])/len(self.dbuffer[key]['data'])).std(0)
      }
      total_mean += latest_stats[key]['mean']
    self.data_stats.append({'by_class': latest_stats, 'full_mean': total_mean/len(self.dbuffer)})
    self.compute_data_drift()

  def add_batch(self, batch, class_label, times_reconstructed = 0):
    """
    Adding a batch to the buffer to the corresponding class storage, 
    The oldest batches are replaced when the buffer is full
    If an unknown class appears - a new storage is added
    Times_reconstructed corresponds to the number of transforms previously applied to the batch: 0 - original data 
    """
    if str(class_label) in self.dbuffer.keys():
      if len(self.dbuffer[str(class_label)]) < self.max_batches_per_class:
        #print('adding new batch')
        self.dbuffer[str(class_label)]['data'].append(batch.clone())
        self.dbuffer[str(class_label)]['times_reconstructed'].append(times_reconstructed)
      else:
        self.dbuffer[str(class_label)]['data'][self.oldest_batches[str(class_label)]] = batch.clone()
        self.dbuffer[str(class_label)]['times_reconstructed'][self.oldest_batches[str(class_label)]] = times_reconstructed
        self.oldest_batches[str(class_label)] = (self.oldest_batches[str(class_label)] + 1) % self.max_batches_per_class
        #print('replacing old batch')
    else:
      #print('Class label:{}'.format(class_label))
      #print('initializing new class')
      self.dbuffer[str(class_label)] = {}
      self.dbuffer[str(class_label)]['data'] = [batch.clone()]
      self.dbuffer[str(class_label)]['times_reconstructed'] = [times_reconstructed]
      self.oldest_batches[str(class_label)] = 0
  
  def load_from_tensor_dataset(self, dataset):
    data_classes = [int(a) for a in set(dataset.tensors[1].tolist())]
    for idx_class in data_classes:
      indices = torch.FloatTensor((dataset.tensors[1].long()==idx_class).tolist()).nonzero().long().squeeze()
      class_loader = DataLoader(dataset, batch_size=self.batch_size, sampler = SubsetRandomSampler(indices),  drop_last=True)
      for (X, Y) in class_loader:
        self.add_batch(X, idx_class)
    if self.compute_stats:
      self.compute_data_stats()
  
  def add_batches_from_dataset(self, dataset, classes_to_add, batches_per_class):
    for idx_class in classes_to_add:
      indices_class = torch.FloatTensor((dataset.tensors[1].long()==idx_class).tolist()).nonzero().long().squeeze()
      class_loader = DataLoader(dataset, batch_size=self.batch_size, sampler = SubsetRandomSampler(indices_class),  drop_last=True)
      for idx_batch in range(batches_per_class):
        self.add_batch(next(iter(class_loader))[0], idx_class)
    print('ADDED DATA FROM THE HISTORICAL DATASET')
    if self.compute_stats:
      self.compute_data_stats()

  def transform_data(self, transform):
    # Inplace apply a given transform to all the batches in the buffer
    transform.eval()
#device = torch.device('cuda:0' if torch.cuda.is_available() and next(transform.parameters()).is_cuda else 'cpu')
    device = torch.device('cuda:{}'.format(self.cuda_device) if torch.cuda.is_available() else 'cpu')
    transformed_buffer = copy.deepcopy(self.dbuffer)
    for class_label in self.dbuffer.keys():
      for idx in range(len(self.dbuffer[str(class_label)]['data'])):
        transformed_buffer[str(class_label)]['data'][idx] = transform(self.dbuffer[str(class_label)]['data'][idx].to(device)).cpu().data
        transformed_buffer[str(class_label)]['times_reconstructed'][idx]+=1
    self.dbuffer = copy.deepcopy(transformed_buffer)
    transform.train()
    if self.compute_stats:
      self.compute_data_stats()
    
  def make_tensor_dataset(self):
    # Transform the buffer into a single tensor dataset
    tensor_data = []
    tensor_labels = []
    tensor_times_reconstructed = []
    for key in self.dbuffer.keys():
      tensor_data += self.dbuffer[key]['data']
      tensor_labels += [int(key)]*(self.batch_size*len(self.dbuffer[key]['data']))
      tensor_times_reconstructed += [times_rec for times_rec in self.dbuffer[key]['times_reconstructed'] for _ in range(self.batch_size)]
    tensor_data = torch.stack(tensor_data)
    tensor_data = tensor_data.reshape(tensor_data.shape[0]*tensor_data.shape[1], tensor_data.shape[2])
    return TensorDataset(tensor_data, torch.FloatTensor(tensor_labels), torch.FloatTensor(tensor_times_reconstructed))
  
  # TESTS
  def test_transforms(self):
    print('TESTING THE CORRECTNESS OF THE CLASS')
    print('Initializing dataset')
    self.dbuffer = {}
    start_time = time.time()
    for idx in range(500):
      self.dbuffer[str(idx)] = [torch.randn(100,32)]*60
    print(self.dbuffer[str(idx)][0].mean())
    print('Initializing took {} seconds'.format(time.time() - start_time))
    tensor_data = self.make_tensor_dataset()
    print('Data reshaped into tensor, taken {} seconds'.format(time.time() - start_time))
    net = nn.Sequential(nn.Linear(32, 512), nn.Linear(512, 32))
    self.transform_data(net)
    print('Data transformed, taken {} seconds'.format(time.time() - start_time))
    print(self.dbuffer[str(idx)][0].mean())
    return tensor_data
  
  def test_batch_adding(self):
    self.dbuffer = {}
    for idx in range(600):
      batch = torch.ones(100, 32)*idx
      class_label = np.random.randint(0, 10)
      self.add_batch(batch, class_label)
  
  def one_batch_test(self):
    self.max_batches_per_class = 1
    for idx in range(600):
      batch = torch.ones(100, 32)*idx
      class_label = np.random.randint(0, 10)
      self.add_batch(batch, class_label)
      
  
  
  
