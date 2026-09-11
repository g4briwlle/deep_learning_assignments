import numpy as np
from scipy.ndimage import label
from skimage.segmentation import watershed


def compute_semantic_metrics(logits, masks_true):
    """
    IoU anf Dice for image batch
    logits: raw output of net (Shape: [B, 1, H, W])
    masks_true: real binary mask (Shape: [B, 1, H, W])
    """
    # make prediction binary (in logits threshold = 0.0)
    preds_bin = (logits > 0.0).float()
    masks_true = masks_true.float()

    # flattens out spatial dimensions
    # shape gets like [B, H*W] to compute each image's metrics separately
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    masks_flat = masks_true.view(masks_true.size(0), -1)

    # true positives
    intersection = (preds_flat * masks_flat).sum(dim=1)

    # union
    union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection

    # IoU and Dice with epsilon to avoid division by zero
    iou = (intersection / (union + 1e-8)).mean().item()
    dice = (2.0 * intersection / (preds_flat.sum(dim=1) + masks_flat.sum(dim=1) + 1e-8)).mean().item()

    return iou, dice


def compute_instance_metrics(pred_mask, real_mask):
    """Compute instance metrics

    pred_mask: 2d matrix with predicted instance mask
    real_mask: 2d matriz with real image instance mask
    """
    # get unique values of instances (ignoring background with value 0)
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0]

    real_ids = np.unique(real_mask)
    real_ids = real_ids[real_ids > 0]

    # absolute count error
    count_error = abs(len(pred_ids) - len(real_ids))

    # build matrix with iou (compare each pred with real value)
    iou_matrix = np.zeros((len(pred_ids), len(real_ids)))

    for i, p_id in enumerate(pred_ids):
        for j, r_id in enumerate(real_ids):
            # calculate intersections (count where prediction and real overlap)
            intersection = np.logical_and(pred_mask == p_id, real_mask == r_id).sum()

            # if there is intersection, calculate union and then iou
            if intersection > 0:
                union = np.logical_or(pred_mask == p_id, real_mask == r_id).sum()
                iou_matrix[i, j] = intersection / union

    # Evaluate TP, FP, FN for each threshold (0.50 to 0.95 with 0.05 step)
    thresholds = np.arange(0.50, 1.0, 0.05)
    aps = []

    for t in thresholds:
        tp = 0
        preds_jointed = set()
        gts_jointed = set()

        # matching rule: greedy by descending iou
        if iou_matrix.size > 0:
            # squeeze matrix and get iou indices from biggest to smallest
            sorted_indices = np.argsort(iou_matrix.flatten())[::-1]

            for idx in sorted_indices:
                iou_value = iou_matrix.flatten()[idx]

                # since it's ordered in descending order, if this iou is smaller than the t, all the next ones will be as well
                if iou_value < t:
                    break

                # get line (pred) and column (real) that generated this iou
                r, c = np.unravel_index(idx, iou_matrix.shape)

                # if none of the both instances was jointed yet, joint them
                if r not in preds_jointed and c not in gts_jointed:
                    tp += 1
                    preds_jointed.add(r)
                    gts_jointed.add(c)

        # false positives are the preds without pair (net hallucination)
        fp = len(pred_ids) - tp

        # false negatives are the real ellipses without pair (blind net)
        fn = len(real_ids) - tp

        # average precision for this t
        ap = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        aps.append(ap)

    # mean average precision from all tested thresholds
    mAP = np.mean(aps)

    return mAP, count_error



