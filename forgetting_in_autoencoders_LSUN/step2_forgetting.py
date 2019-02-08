import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import AE_models

from data_buffer import Data_Buffer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 500
code_size = 32
training_epochs = 100
torch.cuda.set_device(0)
opts = {
  'batch_size': 500,
  'learning_rate': 0.001,
  'betta1': 0.01, # Influence coefficient for classification loss in AE default 1e-2
  'betta2': 3, # Influence coefficient for reconstruction loss in AE
  }

 
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

def test_classifier_on_generator(classif, gen_model, data_loader):
  #TODO add cases of models other than AE
  total = 0
  correct = 0
  gen_model.eval()
  for idx, (test_X,  test_Y) in enumerate(data_loader):
    input_test = gen_model(test_X.float().cuda())
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  gen_model.train()
  return correct/total*100

def get_indices_for_classes(data, data_classes):
  # Creates a list of indices of samples from the dataset, corresponding to given classes
  indices = torch.FloatTensor(list((data.tensors[1].long()==class_).tolist() for class_ in data_classes)).sum(0).nonzero().long().squeeze()
  return indices[torch.randperm(len(indices))]


# Loading the datasets
full_data = torch.load('./data/Synthetic/data_train_test_500_classes_128_features_2000_samples.pth')
trainset = TensorDataset(full_data['data_train'], full_data['labels_train'])
testset = TensorDataset(full_data['data_test'], full_data['labels_test'])

print('Reshaping data into readable format')
prev_classes = list(range(500))
data_buffer = Data_Buffer(4, opts['batch_size'])
for idx_class in prev_classes:
  indices_prev = get_indices_for_classes(trainset, [idx_class])
  prev_loader = DataLoader(trainset, batch_size=opts['batch_size'], sampler = SubsetRandomSampler(indices_prev),  drop_last=True)
  for batch, label in prev_loader:                                                                                
    data_buffer.add_batch(batch.cuda(), idx_class)

print('Ended reshaping')
# Initializing data loaders for first 5 classes

train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier_128_features(nb_of_classes)
classifier_dict = torch.load('./pretrained_models/full_classifier_synthetic_data.pth')
classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = AE_models.AE11(code_size)
gen_model_state = torch.load('./pretrained_models/AE_32_synthetic_data.pth')
gen_model.load_state_dict(gen_model_state)
gen_model.cuda()
#generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion = nn.MSELoss()
#generative_criterion.cuda()

generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
generative_criterion_cl = nn.MSELoss()
generative_criterion_cl.cuda()
generative_criterion_rec = nn.MSELoss()
generative_criterion_rec.cuda()

acc = test_classifier(classifier, test_loader)
acc_rec = test_classifier_on_generator(classifier, gen_model, test_loader)
print('Classification accuracy prior to training: {:.4f}'.format(acc))
print('Test accuracy on reconstructed testset: {}'.format(acc_rec))

accuracies = []
max_accuracy = 0
for epoch in range(training_epochs):  # loop over the dataset multiple times
  if epoch % 1 == 0:
    print('Transforming data with the latest autoencoder')
    data_buffer.transform_data(gen_model)
    trainset = data_buffer.make_tensor_dataset()
    train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X.cuda()
    # ===================forward=====================
    #reconstructions = gen_model(inputs)
    #orig_classes = classifier(inputs)
    #classification_reconstructed = classifier(reconstructions)
    #loss_gen = generative_criterion(classification_reconstructed, orig_classes)
    #loss_gen.backward()
    #generative_optimizer.step()
    #generative_optimizer.zero_grad()
    
    reconstructions = gen_model(inputs)
    orig_classes = classifier(inputs).detach()
    classification_reconstructed = classifier(reconstructions)
    #loss_gen_rec = 0
    loss_gen_rec = opts['betta2']*generative_criterion_rec(reconstructions, inputs)
    loss_gen_cl = opts['betta1']*generative_criterion_cl(classification_reconstructed, orig_classes)
    loss_gen = loss_gen_cl + loss_gen_rec
    loss_gen.backward()
    generative_optimizer.step()
    generative_optimizer.zero_grad()   
    if idx%100==0:
      print('epoch [{}/{}], generators loss: {:.4f}'
        .format(epoch+1, training_epochs,  loss_gen.item()))

    # ===================backward====================
  #acc_test = test_classifier(classifier, test_loader)
  acc_test_rec = test_classifier_on_generator(classifier, gen_model, test_loader)
  accuracies.append(acc_test_rec)
  #print('Test accuracy on trainset: {}'.format(acc_train))
  print('Test accuracy on reconstructed testset: {}'.format(acc_test_rec))
  print('Learning rate for the experiment: {}'.format(opts['learning_rate']))
  if acc_test_rec > max_accuracy:
    max_accuracy = acc_test_rec
    torch.save(gen_model.state_dict(), './pretrained_models/AE_32_synthetic_data_forgetting.pth')
    torch.save(accuracies, './results/forgetting_AE_Synthetic.pth')  
print('End of training, max accuracy {}'.format(max_accuracy))    
print(opts)    
