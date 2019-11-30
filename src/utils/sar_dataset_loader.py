from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        data = np.load(self.dataset_dir + "img_clean_pats.npy", allow_pickle=True)
        self.data_x = data[:,:,:,0]
        self.data_y = data[:,:,:,2]
        print(self.data_x.shape)
        print(self.data_y.shape)
        self.data_x = np.reshape(self.data_x, (self.data_x.shape[0], 1, 572, 572))
        self.data_y = np.reshape(self.data_y, (self.data_x.shape[0], 1, 572, 572))
        print(self.data_x.shape)
    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, i):
        return {'image': torch.from_numpy(self.data_x[i]), 'mask': torch.from_numpy(self.data_y[i])}
