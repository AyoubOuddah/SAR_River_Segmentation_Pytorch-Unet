import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.sar_dataset_loader import BasicDataset
from torch.utils.data import DataLoader, random_split

dataset_dir = './data/dataset/' #execute from drive ==> '/content/drive/My Drive/dataset/'
dir_checkpoint = './model_checkpoints/'


def train_net(net,
              device,
              optimizer,
              criterion,
              e=0,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              train=None,
              val=None):
    if train is None or val is None:
        dataset = BasicDataset(dataset_dir, channels='VVVH', train=True, )
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
    else:
        n_train = len(train)
        n_val = len(val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    for epoch in range(e, epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert true_masks.shape[1] == net.n_classes, \
                    f'Network has been defined with {net.n_classes} output classes, ' \
                    f'but loaded masks have {true_masks.shape[1]} channels. Please check that ' \
                    'the masks are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (
                        (n_train + n_val) // (10 * batch_size)) == 0:  # (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                    writer.add_images('images', (imgs[:, 0, :, :]).view(-1, 1, 572, 572), global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp and ((epoch + 1) % 5 == 0):
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save({
                'epoch': (epoch + 1),
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'train': train,
                'val': val}, dir_checkpoint + "CP_epoch_" + str(epoch + 1) + ".pth")
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-W', '--weights', nargs='+', type=float, default=[1],
                        help='wights used for the imbalenced data', dest='weights')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    myNet = UNet(n_channels=2, n_classes=1)

    logging.info(f'Network:\n'
                 f'\t{myNet.n_channels} input channels\n'
                 f'\t{myNet.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if myNet.bilinear else "Dilated conv"} upscaling')

    myNet.to(device=device)
    # myOptimizer = optim.RMSprop(myNet.parameters(), lr=args.lr, weight_decay=1e-8)
    myOptimizer = optim.Adam(myNet.parameters(), lr=args.lr, eps=1e-08, weight_decay=1e-8)
    if myNet.n_classes > 1:
        myCriterion = nn.CrossEntropyLoss()
    else:
        weights = torch.from_numpy(np.asarray(args.weights)).to(device)
        myCriterion = nn.BCEWithLogitsLoss(weight=weights)
    myEpoch = 0
    myTrainSet = None
    myValSet = None
    if args.load:
        checkpoint = torch.load(args.load)
        myNet.load_state_dict(
            checkpoint['model_state_dict'])
        logging.info(f'Model loaded from {args.load}')
        myOptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f'Optimizer loaded from {args.load}')
        myCriterion = checkpoint['criterion']
        logging.info(f'criterion loaded from {args.load}')
        myEpoch = checkpoint['epoch']
        logging.info(f'current epoch {myEpoch}')
        myTrainSet = checkpoint['train']
        myValSet = checkpoint['val']

    # faster convolutions, but more memory
    #cudnn.benchmark = True

try:
    train_net(net=myNet,
              device=device,
              optimizer=myOptimizer,
              criterion=myCriterion,
              e=myEpoch,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              img_scale=args.scale,
              val_percent=args.val / 100,
              train=myTrainSet,
              val=myValSet)

except KeyboardInterrupt:
    torch.save(myNet.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
