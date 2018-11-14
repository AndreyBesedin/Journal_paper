import numpy as np
import torch
import  torch.distributions.multivariate_normal as mv_n 

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

d = 2048
nb_classes = 10

print('Initializing the matrix')
init_matrix = np.random.rand(d, d)*2-1
print('Performing Gramm-Schmidt')
O = gram_schmidt_columns(init_matrix)
print('Getting random diagonal matrix')
data_class_sampler = {}
class_cov_matrices = {}
class_means = {}

for idx_class in range(nb_classes):
  D = np.array([line*np.random.rand() for line in np.eye(d)])*10
  print('computing unnormalized covariation matrix')
  C = np.matmul(np.matmul(O.transpose(), D), O)

  print('Normalizing covariance matrix')
  diag_C = C.diagonal()
  E = np.eye(d)
  for idx in range(d):
    E[idx][idx]*=(diag_C[idx]**(-1/2))
  class_cov_matrices[idx_class] = np.matmul(np.matmul(E, C), E)
  if not np.allclose(class_cov_matrices[idx_class], class_cov_matrices[idx_class].T, atol=1e-17):
    print('Covarition matrix should be symmetric, break')
    raise('Covarition matrix should be symmetric, break')

  print('Running tests')
  fails = False
  for idx in range(100):
    v = np.random.randn(d)
    res = np.dot(np.matmul(C, v),v)
    if res<=0:
      fails=True
      print(v)
      print(res)
  if not fails:
    print('Tests succesful')

  class_means[idx_class] = torch.zeros(d).double()
  
  data_class_sampler[idx_class] = mv_n.MultivariateNormal(class_means[idx_class], torch.DoubleTensor(class_cov_matrices[idx_class]))


