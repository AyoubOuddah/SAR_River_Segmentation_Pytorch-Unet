import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            mask_pred = net(imgs)
            for idx, true_mask in enumerate(true_masks):
            #Validation with TOP-1 too slow for the training
            #    for i in range(572):
            #        for j in range(572):
            #            if true_mask[0, i, j] == mask_pred[idx, 0, i, j]:
            #                tot += 1
                mask_pred = (mask_pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val
