import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import jaccard_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import multilabel_confusion_matrix

from utils.perMes import *
from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc_score = 0
    recall_score = 0
    jacc_score = 0
    f1_score = 0
    pres_score = 0
    confMat = np.zeros((2,2))
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    acc_score = 0
    recall_score = 0
    f1_score = 0
    pres_score = 0
    confMat = np.zeros((2, 2))
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
                    #pred = np.reshape(pred, -1)
                    pred = pred.astype(bool)
                    true_mask = true_mask.cpu().numpy()
                    #true_mask = np.reshape(true_mask, -1) 
                    true_mask = true_mask.astype(bool)   
                    for i in range(true_mask.shape[0]):
                      for j in range(true_mask.shape[1]):
                        confMat[int(true_mask[0, i, j]), int(pred[0, i, j])] += 1  
                    #confMatrix  = multilabel_confusion_matrix(true_mask, pred)
                    #print(confMatrix.shape)
                   
            pbar.update(imgs.shape[0])

    acc_score = accuracy(confMat)
    pres_score  = precision(confMat)
    recall_score = recall(confMat)
    f1_score = scoreF1(confMat)
    return (tot / n_val), (acc_score / n_val), (pres_score / n_val), (recall_score / n_val), (f1_score / n_val)
                    
