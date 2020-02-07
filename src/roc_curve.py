import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix

import argparse
import logging
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils.sar_dataset_loader import BasicDataset

dataset_dir = './data/dataset/test/'

def compute_roc_curve(net, loader, device, batch_size):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    true_positive_rate = np.zeros((11))
    false_positive_rate = np.zeros((11))
    for i in range(0, 11):
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > (i / 10)).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tn, fp, fn, tp = confusion_matrix(true_mask, pred, labels=[0, 1]).ravel()
                    true_positive_rate[i] += tp / (tp + fn)
                    false_positive_rate[i] += fp / (fp + tn)
        trueDetection[i] /= (len(loader) * batch_size)
        falseAlarm[i] /= (len(loader) * batch_size)
    np.save("true_positive_rate", true_positive_rate)
    np.save("false_positive_rate", false_positive_rate)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    parser.add_argument('--chennel', '-c', type=int,
                        help="number of channels in the patch",
                        default=1)

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    myNet = UNet(n_channels=args.chennel, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    myDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {myDevice}')
    myNet.to(device=myDevice)
    checkpoint = torch.load(args.model)
    myNet.load_state_dict(
        checkpoint['model_state_dict'])

    logging.info("Model loaded !")

    if args.chennel == 1:
        testSet = BasicDataset(dataset_dir, channels='VV', train=False, scale=args.scale)
    else:
        testSet = BasicDataset(dataset_dir, channels='VVVH', train=False, scale=args.scale)

    myLoader = DataLoader(testSet, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True)

    compute_roc_curve(net=myNet, loader=myLoader, device=myDevice, batch_size=args.batchsize)
