import argparse
import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import UNet
from utils.sar_dataset_loader import BasicDataset


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=len(loader), desc='Performance evaluation Score TOP-1', unit='Patch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            mask_pred = net(imgs).cpu().detach().numpy()
            mask_pred = (mask_pred > 0.5)
            correct = (true_masks == mask_pred)
    return np.sum(correct) / len(loader)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path', '-p', default='/content/drive/My Drive/dataset/',
                        metavar='FILE',
                        help="Specify the path to the dataset")

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
        testSet = BasicDataset(args.path, channels='VV', train=False, scale=args.scale)
    else:
        testSet = BasicDataset(args.path, channels='VVVH', train=False, scale=args.scale)

    myLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    eval_net(net=myNet, loader=myLoader, device=myDevice)