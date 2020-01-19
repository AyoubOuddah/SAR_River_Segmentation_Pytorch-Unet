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


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc_score = 0
    rec_score = 0
    f1_score = 0
    pres_score = 0
    jacc_score = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
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
            pbar.update(imgs.shape[0])
    tot = (tot / n_val)
    jacc_score = (jacc_score / n_val)
    acc_score = (acc_score / n_val)
    pres_score = (pres_score / n_val)
    rec_score = (rec_score / n_val)
    if (pres_score + rec_score) > 0:
      f1_score = 2 * (pres_score * rec_score) / (pres_score + rec_score)
    else:
      f1_score = 0
    return tot, jacc_score, acc_score, pres_score, rec_score, f1_score
