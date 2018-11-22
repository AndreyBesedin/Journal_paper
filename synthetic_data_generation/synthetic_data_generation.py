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

def init_means(dim, nb_classes, intra_class_distance, epsilon):    
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

def init_covariances(dim, nb_classes, max_axis, normalized=False):
  print('Initializing the matrix')
  init_matrix = np.random.rand(dim, dim)*2-1
  print('Performing Gramm-Schmidt')
  O = gram_schmidt_columns(init_matrix)
  print('Getting random diagonal matrix')
  class_cov_matrices = {}
  bar = Bar('Initializing the covariances', max=nb_classes)
  for idx_class in range(nb_classes):
    bar.next()
    D = np.array([line*np.random.rand() for line in np.eye(dim)])*max_axis
#    print('computing unnormalized covariation matrix')
    C = np.matmul(np.matmul(O.transpose(), D), O)
    if normalized:
#      print('Normalizing covariance matrix')
      diag_C = C.diagonal()
      E = np.eye(dim)
      for idx in range(dim):
        E[idx][idx]*=(diag_C[idx]**(-1/2))
      class_cov_matrices[idx_class] = np.matmul(np.matmul(E, C), E)
    else:
      class_cov_matrices[idx_class] = C
    if not np.allclose(class_cov_matrices[idx_class], class_cov_matrices[idx_class].T, atol=1e-17):
      print('Covarition matrix should be symmetric, break')
      raise('Covarition matrix should be symmetric, break')
  bar.finish()
  return class_cov_matrices

def initialize_synthetic_sampler(dim, nb_classes, intra_class_distance=10, max_axis=1, epsilon=1e-3):
  means_ = init_means(dim, nb_classes, intra_class_distance, epsilon)
  covariances = init_covariances(dim, nb_classes, max_axis, False)
  data_class_sampler = {}
  for idx_class in range(nb_classes):
    data_class_sampler[idx_class] = mv_n.MultivariateNormal(torch.DoubleTensor(means_[idx_class]), torch.DoubleTensor(covariances[idx_class]))
  return data_class_sampler
    
def sample_data_from_sampler(data_sampler, samples_per_class):
  feature_size = data_sampler[0].sample().shape[0]
  nb_of_classes = len(data_sampler) 
  data_size = nb_of_classes*samples_per_class
  data_ = torch.zeros(data_size, feature_size)
  labels_ = torch.zeros(data_size)
  bar = Bar('Generating data ', max=nb_of_classes)
  for idx_class in range(nb_of_classes):
    bar.next()
    data_[idx_class*samples_per_class:(idx_class+1)*samples_per_class] = data_sampler[idx_class].sample((samples_per_class,))
    labels_[idx_class*samples_per_class:(idx_class+1)*samples_per_class] = idx_class
  bar.finish()
  return (data_, labels_)

def sample_big_data(feature_size, nb_of_classes, samples_per_class):
  full_data = {}
  data_size = nb_of_classes*samples_per_class
  
  full_data['means_'] = init_means(feature_size, nb_of_classes, 1.7, 1e-3)
  full_data['covariances'] = init_covariances(feature_size, nb_of_classes, 1)
  full_data['data_train'] = torch.zeros(data_size, feature_size)
  full_data['data_test'] = torch.zeros(data_size, feature_size)
  full_data['labels_train'] = torch.zeros(data_size)
  full_data['labels_test'] = torch.zeros(data_size)
  
  bar = Bar('Generating data ', max=nb_of_classes)
  for idx_class in range(nb_of_classes):
    bar.next()
    data_sampler = mv_n.MultivariateNormal(torch.DoubleTensor(full_data['means_'][idx_class]), torch.DoubleTensor(full_data['covariances'][idx_class]))
    full_data['data_train'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = data_sampler.sample((samples_per_class,))
    full_data['data_test'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = data_sampler.sample((samples_per_class,))    
    full_data['labels_train'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = idx_class
    full_data['labels_test'][idx_class*samples_per_class:(idx_class+1)*samples_per_class] = idx_class
  bar.finish()
  return full_data

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#for classes in range(nb_classes):
  #color = tuple(np.random.rand(3))
  #for idx in range(50):
    #p = data_class_sampler[classes].sample().numpy()
    #ax.plot([p[0]], [p[1]], [p[2]], 'o', color = color)
    
#for classes in range(nb_classes):
  #color = tuple(np.random.rand(3))
  #for idx in range(100):
    #p = data_class_sampler[classes].sample().numpy()
    #plt.plot(p[0], p[1], 'o', color = color)
    

#plt.show()
