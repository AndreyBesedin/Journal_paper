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
  
  def load_from_tensor_dataset(self, dataset):
    nb_of_classes = int(dataset.tensors[1].max().item()) + 1
    for idx_class in range(nb_of_classes):
      indices = torch.FloatTensor(list((dataset.tensors[1].long()==idx_class).tolist())).nonzero().long().squeeze()
      class_loader = DataLoader(dataset, batch_size=self.batch_size, sampler = SubsetRandomSampler(indices),  drop_last=True)
      for (X, Y) in class_loader:
        self.add_batch(X, idx_class)
      
  def transform_data(self, transform):
    # Inplace apply a given transform to all the batches in the buffer
    device = torch.device('cuda:0' if torch.cuda.is_available() and next(transform.parameters()).is_cuda else 'cpu')
    transformed_buffer = copy.deepcopy(self.dbuffer)
    for class_label in self.dbuffer.keys():
      for idx in range(len(self.dbuffer[str(class_label)])):
        transformed_buffer[str(class_label)][idx] = transform(self.dbuffer[str(class_label)][idx].to(device)).data
    self.dbuffer = copy.deepcopy(transformed_buffer)
        
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
  
  # TEST
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
      
  
  
  
