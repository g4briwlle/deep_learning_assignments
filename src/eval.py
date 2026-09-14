import numpy as np
from scipy.ndimage import label
import torch

from .metrics import *
from .utils import extract_instances_watershed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def part1_eval(model, val_loader):
    model.eval()

    mAPs_inst = []
    erros_contagem_inst = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)

            # real instances id
            masks_gt_np = masks.cpu().numpy()

            # trained net make predictions
            logits = model(images)

            # turn it into prob and binaryze it
            probs = torch.sigmoid(logits)
            preds_binary_np = (probs > 0.5).cpu().numpy().squeeze(1)

            # extract instances and compute metrics image by image from batch
            for i in range(images.size(0)):
                # part 1.2: label connected components
                pred_instance_mask, _ = label(preds_binary_np[i])
                real_instance_mask = masks_gt_np[i]

                # part 1.3: call instance metrics func
                mAP_img, erro_img = compute_instance_metrics(pred_instance_mask, real_instance_mask)

                mAPs_inst.append(mAP_img)
                erros_contagem_inst.append(erro_img)

    # final means from eval set
    mAP_final = np.mean(mAPs_inst)
    erro_final = np.mean(erros_contagem_inst)

    print(f"mAP (IoU 0.50 to 0.95): {mAP_final:.4f}")
    print(f"mean of absolute count error: {erro_final:.4f} objects/image")

def part2_eval(model, val_loader):
    model.eval()

    mAPs_inst = []
    erros_contagem_inst = []

    with torch.no_grad():
        for images, _, masks_instance_gt in val_loader:
            images = images.to(device)
            # answer sheet
            masks_gt_np = masks_instance_gt.cpu().numpy()

            # net predicts 3 channels
            logits = model(images)

            # process image by image inside batch
            for i in range(images.size(0)):

                # pos processing trail A
                # pass output [3, H, W] of a single image to watershed
                pred_instance_mask = extract_instances_watershed(logits[i])
                real_instance_mask = masks_gt_np[i]

                # metrics calculation
                mAP_img, erro_img = compute_instance_metrics(pred_instance_mask, real_instance_mask)

                mAPs_inst.append(mAP_img)
                erros_contagem_inst.append(erro_img)

    # final results for epoch
    map_final = np.mean(mAPs_inst)
    erro_final = np.mean(erros_contagem_inst)

    print(f"mAP (IoU 0.50 to 0.95): {map_final:.4f}")
    print(f"mean of absolute count error: {erro_final:.4f} objects/image")