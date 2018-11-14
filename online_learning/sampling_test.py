import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

root = '/home/besedin/workspace/Projects/Journal_paper/'
train_dataset = datasets.MNIST(root=root+'datasets/',
  train=True,
  download=True,
  transform=img_transform)
test_dataset = datasets.MNIST(root=root+'datasets/',
  train=False,
  download=True,
  transform=img_transform)

test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
#train_dataset_ = torch.load('../datasets/LSUN_features/testset.pt')
#train_dataset = TensorDataset(train_dataset_[0], train_dataset_[1])
for idx in range(10):
  data_class = torch.randint(0, 10, (1,))
  indices = (train_dataset.train_labels.long()==data_class.long()).nonzero().long()
  train_loader = DataLoader(train_dataset, batch_size=1000, sampler=SubsetRandomSampler(indices.squeeze()))
  for idx_batch, data in enumerate(train_loader):
    print(data[1])
    if idx_batch>10:
      break
