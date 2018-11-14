import numpy as np 
import pickle as pkl 







data_root = '/home/besedin/workspace/Projects/Journal_paper/datasets/original_data/CORe50/'
pkl_file = open(data_root + 'paths.pkl', 'rb') 
paths = pkl.load(pkl_file) 
imgs = np.load(data_root+'core50_imgs.npz')['x'] 
