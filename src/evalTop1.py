import argparse
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import UNet
from utils.sar_dataset_loader import BasicDataset
from utils.perMes import *

dataset_dir = './data/dataset/'  # execute from drive ==> '/content/drive/My Drive/dataset/'


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    confMat = np.zeros((2, 2))  # make it n_class X n_class to generate
    # confMat.to(device=device, dtype=torch.float64)
    # with tqdm(total=len(loader), desc='Performance evaluation Score TOP-1', unit='Patch', leave=False) as pbar:
    for batch in loader:
        print("------- new batch!! --------")
        imgs = batch['image']
        true_masks = batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        # true_masks = true_masks.to(device=device, dtype=torch.float32)
        mask_pred = net(imgs)
        # mask_pred = mask_pred.detach().cpu().numpy()
        mask_pred = (mask_pred > 0.5).int().detach().cpu().numpy()
        # torch.cuda.synchronize()
        for idx in range(true_masks.shape[0]):
            for i in range(true_masks.shape[2]):
                for j in range(true_masks.shape[3]):
                    confMat[int(true_masks[idx, 0, i, j].item()), int(mask_pred[idx, 0, i, j].item())] += 1
                    tot += 1
    acc = accuracy(confMat)
    rec = recall(confMat)
    pre = precision(confMat)
    sco = scoreF1(confMat)

    # confMat = confMat / (len(loader) * 572 * 572 * 4)
    print("conf matrix:")
    print(confMat)
    print("sum ==> ", np.sum(confMat))
    print("total ==> ", tot)
    print("acc ==> ", acc)
    print("rec ==> ", rec)
    print("pre ==> ", pre)
    print("sco ==> ", sco)
    return confMat, acc, rec, pre, sco

    # correct = (true_masks == mask_pred)
    # tot += torch.sum(correct).item()
    # return tot / (len(loader) * 572 * 572 * 4)

    #        mask_pred = net(imgs).cpu().detach().numpy()
    #        mask_pred = (mask_pred > 0.5)
    #        correct = (true_masks == mask_pred)
    # return np.sum(correct) / len(loader)


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

    eval_net(net=myNet, loader=myLoader, device=myDevice)
