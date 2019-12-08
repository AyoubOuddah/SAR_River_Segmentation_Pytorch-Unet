import argparse
import glob
import random
import os
import numpy as np
from skimage.transform import resize

basedir = '../data'

DATA_AUG_TIMES = 8  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default="%s/Nor_DATA" % basedir, help='dir of data') # check
parser.add_argument('--save_dir', dest='save_dir', default="%s/dataset/test/" % basedir, help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=572, help='patch size') # check
parser.add_argument('--stride', dest='stride', type=int, default=286, help='stride') # check
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=4, help='batch size') # check
# check output arguments
parser.add_argument('--from_file', dest='from_file', default="%s/data/img_clean_pats.npy" % basedir, help='get pic from file')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
args = parser.parse_args()

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.npy')
    if isDebug:
        filepaths = filepaths[:10]
    print("number of training data %d" % len(filepaths))

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = np.load(filepaths[i])
        im_h = np.size(img, 0)
        im_w = np.size(img, 1)
        if im_h < args.pat_size:
          img = resize(img, (args.pat_size+1, im_w))
          im_h = np.size(img, 0)

        if im_w < args.pat_size:
          img = resize(img, (im_w, args.pat_size+1))
          im_w = np.size(img, 1)

        for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
            for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                count += 1
    origin_patch_num = count * DATA_AUG_TIMES

    if origin_patch_num % args.bat_size != 0:
        numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    else:
        numPatches = origin_patch_num
    print("total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size))

    # data matrix 4-D
    numPatches=int(numPatches)
    input = np.zeros((1, args.pat_size, args.pat_size, 3), dtype="float32")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    count = 0
    # generate patches
    for i in range(len(filepaths)): #prima scorre le immagini
        img = np.load(filepaths[i])
        im_h = np.size(img, 0)
        im_w = np.size(img, 1)
        if im_h < args.pat_size:
          img = resize(img, (args.pat_size+1, im_w))
          im_h = np.size(img, 0)

        if im_w < args.pat_size:
          img = resize(img, (im_w, args.pat_size+1))
          im_w = np.size(img, 1)
        img_s = img
        img_s = np.reshape(np.array(img_s, dtype="float32"),
                              (np.size(img_s, 0), np.size(img_s, 1), 3))  # extend one dimension
        for j in range(DATA_AUG_TIMES):
            im_h = np.size(img, 0)
            im_w = np.size(img, 1)
            if DATA_AUG_TIMES == 8:
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        input = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                              j)
                        count += 1
                        file_name, _ = os.path.splitext(filepaths[i])
                        np.save(args.save_dir+"patch_"+str(count)+".npy", np.array(input))
            else:
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):

                        input = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                      random.randint(0, 7))
                        count += 1
                        file_name, _ = os.path.splitext(filepaths[i])
                        np.save(args.save_dir+"patch_"+str(count)+".npy", np.array(input))
    # pad the batch
    #if count < numPatches:
    #    to_pad = numPatches - count
    #    inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    #if not os.path.exists(args.save_dir):
    #    os.mkdir(args.save_dir)
    #np.save(os.path.join(args.save_dir, "Test3"), inputs)
    print(count)

if __name__ == '__main__':
    generate_patches()
