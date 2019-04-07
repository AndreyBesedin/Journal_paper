import matplotlib.pyplot as plt
import torch


def mean(l):
  return sum(l)/len(l)

def reduced(l, s):
  return [l[0]] + [mean(l[(idx)*s:(idx+1)*s]) for idx in range(int(len(l)/s))]

pure_stream = torch.load('Synthetic_stream_0_fake_batches_0_hist_batches_2_batches_in_storage_0.3_betta1_0.1_betta2.pth')
stream_small_rehearsal = torch.load('Synthetic_stream_0_fake_batches_15_hist_batches_5_batches_in_storage_0.3_betta1_0.1_betta2.pth')
#stream_with_no_rec_counter = torch.load('LSUN_stream_15_fake_batches_5_hist_batches_20_batches_in_storage_0.1_betta1_1_betta2_no_rec_counter.pth')
#stream_with_only_gen_rec_counter = torch.load('LSUN_stream_15_fake_batches_5_hist_batches_20_batches_in_storage_0.1_betta1_1_betta2_only_use_rec_counter_for_generator.pth')
stream_full_approach = torch.load('Synthetic_stream_10_fake_batches_10_hist_batches_10_batches_in_storage_0.3_betta1_0.1_betta2.pth') 
stream_only_rec_loss = torch.load('Synthetic_stream_10_fake_batches_10_hist_batches_10_batches_in_storage_0_betta1_0.1_betta2.pth')
stream_only_cl_loss =  torch.load('Synthetic_stream_10_fake_batches_10_hist_batches_10_batches_in_storage_0.3_betta1_0_betta2.pth')

pure_stream = pure_stream['accuracies'][:400]
stream_small_rehearsal = stream_small_rehearsal['accuracies'][:400]
stream_full_approach = stream_full_approach['accuracies'][:400]
stream_only_rec_loss = stream_only_rec_loss['accuracies'][:400]
stream_only_cl_loss = stream_only_cl_loss['accuracies'][:400]

#plt.plot(stream_small_rehearsal, label='stream_small_rehearsal')
#plt.plot(pure_stream, label='pure_stream')
#plt.plot(stream_full_approach, label='stream_full_approach')
#plt.plot(stream_only_rec_loss, label='stream_only_rec_loss')
#plt.plot(stream_only_cl_loss, label='stream_only_cl_loss')

s=10
x = [0]+[idx*s for idx in range(1, 41)]
plt.plot(x, [96.1]*41, 'r--', label='offline training')

plt.plot(x, reduced(pure_stream, s), label='pure stream', marker='.')
plt.plot(x, reduced(stream_small_rehearsal, s), label='stream + reh', marker='o')
plt.plot(x, reduced(stream_full_approach, s), label='stream + reh + pseudo-reh(beta1=1, beta2=0)', marker='v')
plt.plot(x, reduced(stream_only_cl_loss, s), label='stream + reh + pseudo-reh(beta1=0, beta2=0.1)', marker='1')
plt.plot(x, reduced(stream_only_rec_loss, s), label='stream + reh + pseudo-reh(beta1=1, beta2=0.1)', marker='*')
plt.xlabel('Training interval')
plt.ylabel('Classification accuracy')
plt.legend()
plt.grid()
plt.show()
