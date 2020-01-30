import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from dice_loss import dice_coeff
from sklearn.metrics import confusion_matrix

import argparse
import logging
import os
from torch.utils.data import DataLoader
from unet import UNet
from utils.sar_dataset_loader import BasicDataset

dataset_dir = '/content/drive/My Drive/dataset/'


def eval_net(net, loader, device, batch_size):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc_score = 0
    rec_score = 0
    f1_score = 0
    pres_score = 0
    jacc_score = 0

    trueDetection = np.zeros((10))
    falseAlarm = np.zeros((10))
    index  = 0
    for seuill in range(0.1, 1, 0.1):
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                    pred = pred.detach().cpu().numpy()
                    pred = pred.astype(int)
                    pred = np.matrix.flatten(pred)

                    true_mask = true_mask.cpu().numpy()
                    true_mask = true_mask.astype(int)
                    true_mask = np.matrix.flatten(true_mask)

                    jacc_score += jaccard_score(true_mask, pred)
                    acc_score += accuracy_score(true_mask, pred)
                    pres_score += precision_score(true_mask, pred)
                    rec_score += recall_score(true_mask, pred)
                    tn, fp, fn, tp = confusion_matrix(true_mask, pred, labels=[0, 1]).ravel()
                    sum = np.sum(true_mask)
                    trueDetection[i] += tp/sum
                    falseAlarm[i] += (tp + fp) / sum
        trueDetection[i] /= (len(loader) * batch_size)
        falseAlarm[i] /= (len(loader) * batch_size)
        tot = (tot / (len(loader) * batch_size))
        jacc_score = (jacc_score / (len(loader) * batch_size))
        acc_score = (acc_score / (len(loader) * batch_size))
        pres_score = (pres_score / (len(loader) * batch_size))
        rec_score = (rec_score / (len(loader) * batch_size))
        if (pres_score + rec_score) > 0:
            f1_score = 2 * (pres_score * rec_score) / (pres_score + rec_score)
        else:
            f1_score = 0

        print("Seuill :  ", seuill)
        print("Dice : ", tot)
        print("Jaccard: ", jacc_score)
        print("Accuracy: ", acc_score)
        print("Pres :", pres_score)
        print("Recall: ", rec_score)
        print("F1_score ", f1_score)
    np.save("trueDetection.py", trueDetection)
    np.save("falseDetection.py", falseDetection)
    return tot, jacc_score, acc_score, pres_score, rec_score, f1_score
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

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

    eval_net(net=myNet, loader=myLoader, device=myDevice, batch_size=args.batchsize)
