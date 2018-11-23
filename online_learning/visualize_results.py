import matplotlib.pyplot as plt
import torch

def get_accuracy(conf_list):
  accuracies = []
  for idx_epoch in range(len(conf_list)):
    res = 0
    for idx_class in range(10):
      res+=conf_list[idx_epoch][idx_class][idx_class]/10
    accuracies.append(res)
  return accuracies

code_size = {2, 4, 8, 16, 32}
colors = ['b', 'g', 'r', 'c', 'm']
common_name = 'results/train_progress_forgetting_ae_'
accuracies = []
for idx, scenario in enumerate(('', '_with_real')):
  for idx_code, cs in enumerate(code_size):
    filename = common_name+str(cs)+scenario+'.pth'
    res = torch.load(filename)
    acc = get_accuracy(res['confusion'])
    accuracies.append(acc)
    if idx==1:
      plt.plot(range(525), acc, color=colors[idx_code], linestyle='-.', label='code_size-'+str(cs)+scenario)
    else:
      plt.plot(range(525), acc, color=colors[idx_code], label='code_size-'+str(cs)+scenario)
    
plt.plot(range(525), [96.2]*525, linestyle='--',label='trained on 5% of data')
plt.grid(b=None, which='major', axis='both')
plt.legend()
plt.show()
