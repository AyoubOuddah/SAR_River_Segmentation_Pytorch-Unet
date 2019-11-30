from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, root_dir = "./data/dataset/", channel = 'VV' train = True):
        self.dataset_dir = root_dir
        self.channel = channel
        if Train:
            self.dataset_dir += "train/"
        else:
            self.dataset_dir += "train/"
        self.ids = [splitext(file)[0] for file in listdir(root_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        patch_file = glob(self.root_dir + idx  + '*')
        assert len(patch_file) == 1, \
            f'Either no patch or multiple patchs found for the ID {idx}: {data_file}'
        patch = np.load(patch_file, allow_pickle=True)
        mask = np.reshape(patch[:,:,2],(1,572,572))
        img = np.reshape(patch[:,:,0],(1,572,572))
        if self.channel = 'VH':
            img = np.reshape(data[:,:,1],(1,572,572))
        elif self.channel = 'VVVH':
            img = np.reshape(data[:,:,0:2],(2,572,572))
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
