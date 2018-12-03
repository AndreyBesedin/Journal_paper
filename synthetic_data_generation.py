import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.distributions.multivariate_normal as mv_n
#from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar
def dist(v1, v2):
  return np.dot(v1-v2, v1-v2)

def dist_to_set(vect, set_of_vect):
  dist_min = dist(vect, set_of_vect[0])
  for vect_old in set_of_vect:
    dist_min = min(dist_min, dist(vect, vect_old))
  return dist_min

def init_means(dim, nb_classes, intra_class_distance, epsilon=1e-4):    
  res = [np.zeros(dim)]
  bar = Bar('Initializing the means:', max=nb_classes)
  for idx_class in range(1, nb_classes):
    bar.next()
    v_new = np.random.randn(dim)/100
    while dist_to_set(v_new, res)<intra_class_distance:
      for idx_prev in range(idx_class):
        dist_ = dist(v_new, res[idx_prev])
        if dist_< intra_class_distance:
          v_new = res[idx_prev] + (v_new - res[idx_prev])*(intra_class_distance + epsilon)/dist_
    res.append(v_new)
  bar.finish()
  return np.array(res)

def gram_schmidt(vectors):
  basis = []
  for v in vectors:
    w = v - np.sum( np.dot(v,b)*b  for b in basis )
    if (w > 1e-10).any():
      basis.append(w/np.linalg.norm(w))
  return np.array(basis)

def gram_schmidt_columns(X):
  Q, R = np.linalg.qr(X)
  return Q

def compute_covariance(transform_matrix, D):
  return np.matmul(np.matmul(transform_matrix.transpose(), np.diag(D)), transform_matrix)
  
def init_covariances(dim, nb_classes, max_axis):
  print('Initializing the matrix')
  init_matrix = np.random.rand(dim, dim)*2-1
  print('Performing Gramm-Schmidt')
  transform_matrix = gram_schmidt_columns(init_matrix)
  D_vectors = {}
  bar = Bar('Initializing the covariances', max=nb_classes)
  for idx_class in range(nb_classes):
    bar.next()
    D = np.array([line*np.random.rand() for line in np.eye(dim)])
    D=D/D.max()*max_axis
    D_vectors[idx_class] = D.diagonal()
    C = np.matmul(np.matmul(transform_matrix.transpose(), D), transform_matrix)
    if not np.allclose(C, C.T, atol=1e-17):
      raise('Covarition matrix should be symmetric, break')
  bar.finish()
  return transform_matrix, D_vectors

def sample_class(mean_, covariance_transform_, diagonal_, nb_of_samples_):
  covariance = compute_covariance(covariance_transform_, diagonal_)
  data_sampler = mv_n.MultivariateNormal(torch.DoubleTensor(mean_), torch.DoubleTensor(covariance))
  return data_sampler.sample((nb_of_samples_,))
  
def sample_big_data(feature_size, nb_of_classes, samples_per_class, inter_class_distance=2):
  full_data = {}
  data_size = nb_of_classes*samples_per_class
  
  full_data['means_'] = init_means(feature_size, nb_of_classes, inter_class_distance)
  full_data['covariance_transform'], full_data['covariance_diagonals'] = init_covariances(feature_size, nb_of_classes, 1)
  full_data['data_train'] = torch.zeros(data_size, feature_size)
  full_data['data_test'] = torch.zeros(data_size, feature_size)
  full_data['labels_train'] = torch.zeros(data_size)
  full_data['labels_test'] = torch.zeros(data_size)
  
  bar = Bar('Generating data ', max=nb_of_classes)
  for idx_class in range(nb_of_classes):
    bar.next()
    covariance = compute_covariance(full_data['covariance_transform'], full_data['covariance_diagonals'][idx_class])
    data_sampler = mv_n.MultivariateNormal(torch.DoubleTensor(full_data['means_'][idx_class]), torch.DoubleTensor(covariance))
    full_data['data_train'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = data_sampler.sample((samples_per_class,))
    full_data['data_test'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = data_sampler.sample((samples_per_class,))    
    full_data['labels_train'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = idx_class
    full_data['labels_test'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = idx_class
  bar.finish()
  return full_data

