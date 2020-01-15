import torch
import torch.nn.functional as F
from tqdm import tqdm
import sklearn
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
    #confMatrix = np.zeros((2,2))
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.float32)
            mask_pred = net(imgs)
            mask_pred = (mask_pred > 0.5).int()
            for idx, true_mask in enumerate(true_masks):
                mask_pred = (mask_pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(mask_pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tot += dice_coeff(mask_pred, true_mask.squeeze(dim=1)).item()

            mask_pred = mask_pred.detach().cpu().numpy()
            true_mask = mask_pred.cpu().numpy()
            print(mask_pred.shape)
            print(true_mask.shape)
            acc_score += sklearn.metrics.accuracy_score(true_mask, mask_pred)
            pres_score += sklearn.metrics.precision_score(true_mask, mask_pred)
            jacc_score += sklearn.metrics.jaccard_score(true_mask, mask_pred)
            recall_score += sklearn.metrics.recall_score(true_mask, mask_pred)
            f1_score += sklearn.metrics.f1_score(true_mask, mask_pred)

            pbar.update(imgs.shape[0])

    return (tot / n_val), (acc_score / n_val), (pres_score / n_val), (jacc_score/ n_val), (recall_score / n_val), (f1_score / n_val)
