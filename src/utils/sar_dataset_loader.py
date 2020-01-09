from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from skimage.transform import resize

class BasicDataset(Dataset):
    def __init__(self, root_dir = "../data/dataset/", channels = 'VV', train = True, scale=1):
        self.dataset_dir = root_dir
        self.channels = channels
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        if train:
            self.dataset_dir += "train/"
        else:
            self.dataset_dir += "test/"
        try:
            self.ids = [splitext(file)[0] for file in listdir(self.dataset_dir)
                    if not file.startswith('.')]
            logging.info(f'Creating dataset with {len(self.ids)} examples')
        except FileNotFoundError:
            print("Wrong file or file path")
    def __len__(self):
        return len(self.ids)

    def preprocess(self, patch):
        h = patch.shape[0]
        w = patch.shape[1]
        newW, newH = int(self.scale * w), int(self.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        patch = resize(patch, (newH, newW))

        if len(patch.shape) == 2:
            patch = np.expand_dims(patch, axis=2)

        # HWC to CHW
        patch = patch.transpose((2, 0, 1))

        return patch

    def __getitem__(self, i):
        idx = self.ids[i]
        patch_file = self.dataset_dir + idx  + '.npy'

        patch = np.load(patch_file, allow_pickle=True)

        mask = patch[:,:,2]
        img = patch[:,:,0]
        if self.channels == 'VH':
            img = patch[:,:,1]
        elif self.channels == 'VVVH':
            img = patch[:,:,0:2]
        img = self.preprocess(img)
        mask = self.preprocess(mask)
        #mask = np.reshape(patch[:,:,2],(1,572,572))
        #img = np.reshape(patch[:,:,0],(1,572,572))
        #if self.channel == 'VH':
         #   img = np.reshape(data[:,:,1],(1,572,572))
        #elif self.channel == 'VVVH':
         #   img = np.reshape(data[:,:,0:2],(2,572,572))
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
