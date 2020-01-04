import os
import scipy.io as sio
from tqdm import tqdm
import numpy as np
#from sklearn.feature_extraction import image

class SarDataset():
    #PATCH_SIZE = 32
    INPUT_PATH = "../data/Raw_DATA/"#"./drive/My Drive/Raw_DATA"
    EXTRACTION_PATH = "../data/Ext_DATA/"
    NORMALIZATION_PATH = "../data/Nor_DATA/"
    DATASET_PATH = "/content/drive/My Drive/dataset/train/"
    data_counter = 0

    mean_vv = 0
    mean_vh = 0

    std_vv = 0
    std_vh = 0

    def extract_dataSet_images(self):
        for f in tqdm(os.listdir(self.INPUT_PATH)):
            if "mat" in f:
                try:
                    path = os.path.join(self.INPUT_PATH, f)
                    sarData = sio.loadmat(path)
                    for num_data in range(1): #range(sarData["array_segmentation"].shape[2]):
                        img_vv = sarData["images_choisiesVV"][:,:,num_data]
                        img_vh = sarData["images_choisiesVH"][:,:,num_data]
                        img_gt = sarData["array_segmentation"][:,:,num_data]
                        self.mean_vv += img_vv.mean()
                        self.std_vv += img_vv.std()
                        self.mean_vh += img_vh.mean()
                        self.std_vh += img_vh.std()
                        self.data_counter += 1
                        file_name, _ = os.path.splitext(f)
                        data_to_save = np.stack((np.array(img_vv), np.array(img_vh), np.array(img_gt)), axis=2)
                        np.save(self.EXTRACTION_PATH + file_name + str(num_data) + "_.npy", data_to_save)
                except Exception as e:
                    print(e)
        self.mean_vv /= self.data_counter
        self.mean_vh /= self.data_counter
        self.std_vv /= self.data_counter
        self.std_vh /= self.data_counter
        print("Done!")
        print("data_counter = ", self.data_counter)
        print("mean_vv = ", self.mean_vv)
        print("mean_vh = ", self.mean_vh)
        print("std_vv = ", self.std_vv)
        print("std_vh = ", self.std_vh)

    def normalize_dataSet_images(self):
        for f in tqdm(os.listdir(self.EXTRACTION_PATH)):
            if "npy" in f:
                try:
                    path = os.path.join(self.EXTRACTION_PATH, f)
                    data = np.load(path)
                    data_vv = data[:,:,0]
                    data_vh = data[:,:,1]
                    data_vv[data_vv > (self.mean_vv + (3 * self.std_vv))] = (self.mean_vv + (3 * self.std_vv))
                    data_vh[data_vh > (self.mean_vh + (3 * self.std_vh))] = (self.mean_vh + (3 * self.std_vh))
                    data_vv = data_vv / (self.mean_vv + (3 * self.std_vv))
                    data_vh = data_vh / (self.mean_vh + (3 * self.std_vh))
                    file_name, _ = os.path.splitext(f)
                    data_to_save = np.stack((np.array(data_vv), np.array(data_vv), np.array(data[:,:,2])), axis=2)
                    np.save(self.NORMALIZATION_PATH + file_name + ".npy", data_to_save)
                except Exception as e:
                    print(e)
    
    def compute_dataSet_wight(self):
        patch_counter = 0 
        fg_pixel_counter = 0
        for f in tqdm(os.listdir(self.DATASET_PATH)):
            if "npy" in f:
                try:
                    path = os.path.join(self.DATASET_PATH, f)
                    data = np.load(path)
                    mask = data[:,:,2]
                    fg_pixel_counter += np.sum(mask)
                    patch_counter += 1
                except Exception as e:
                    print(e)
        return (((patch_counter * 572 * 572) - fg_pixel_counter) / fg_pixel_counter)
