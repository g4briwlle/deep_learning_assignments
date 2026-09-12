"""Modulo of util functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.segmentation import watershed

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

from .metrics import *

import torch

def extract_instances_naive(logits_tensor, threshold = 0.5):
    """ Transform one channel outpuf of net into instance mask with unique ids
    logits_tensor: model output in [1, H, W]
    threshold: probability cut-off
    """
    # apply sigmoid in logit to go from (-inf, +inf) to (0, 1)
    probs = torch.sigmoid(logits_tensor)

    # apply threshold to get binary mask
    binary_mask = (probs > threshold).cpu().numpy().squeeze()

    # connected components: gives each ellipse an id
    instance_mask, n_objects = label(binary_mask)

    return instance_mask, n_objects



def plot_density_failure(model, val_loader, device):
    model.eval()

    # list to store tuples ( qtd_real_objs, count_error)
    plot_data = []

    print("Evaluating images and grouping densities...")
    with torch.no_grad():
        for images, masks_gt in val_loader:
            images = images.to(device)
            masks_gt_np = masks_gt.cpu().numpy()

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_binary_np = (probs > 0.5).cpu().numpy().squeeze(1)

            for i in range(images.size(0)):
                pred_instance_mask, _ = label(preds_binary_np[i])
                real_instance_mask = masks_gt_np[i]

                # get error from the instance metrics function
                _, erro_img = compute_instance_metrics(pred_instance_mask, real_instance_mask)

                # count how many real objects (density) there are in this specific image
                real_ids = np.unique(real_instance_mask)
                num_objetos_reais = len(real_ids[real_ids > 0])

                plot_data.append((num_objetos_reais, erro_img))

    # extract data to np arrays
    densidades = np.array([d[0] for d in plot_data])
    erros = np.array([d[1] for d in plot_data])

    # group and compute mean error for each specific density (n of ellipses)
    densidades_unicas = np.sort(np.unique(densidades))
    erros_medios = [np.mean(erros[densidades == d]) for d in densidades_unicas]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(densidades_unicas, erros_medios, marker='o', linestyle='-', color='red', linewidth=2)

    plt.title("Parte 1.5: Aumento do Erro de Contagem pela Densidade (Método Ingênuo)", fontsize=14)
    plt.xlabel("Densidade de Objetos (Qtd. de Elipses Reais)", fontsize=12)
    plt.ylabel("Erro Absoluto de Contagem Médio", fontsize=12)
    plt.xticks(densidades_unicas)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()




def extract_instances_watershed(logits_tensor):
    """
    Transform net 3d output (bg, inside, border) in isoladed instance
    logits_tensor: raw net output [3, H, W]
    """
    # transform logits in multiclass probs
    # softmax in 0 axis (3 class one)
    probs = torch.softmax(logits_tensor, dim=0)

    # get winning class for each pixel, matrix gets to be [H, W] cointaing the values 0, 1 or 2
    pred_class = torch.argmax(probs, dim=0).cpu().numpy()

    # separate classes
    inside = (pred_class == 1)
    border = (pred_class == 2)

    # area where the water can spread
    foreground = (inside | border)

    # water sources
    # the label function finds the inside isles separately and give each an id
    markers, _ = label(inside)

    # elevation map
    # the water doesnt spread beyond the border
    elevation_map = border.astype(float)

    # apply watershed
    # water comes from markers, elevates elevation map and stops in the masks
    instance_mask = watershed(elevation_map, markers, mask=foreground)

    return instance_mask


class FocalLossMulticlass(nn.Module):
    """
    Focal Loss for 3 classes (bg, inside, border)
    CE standard -> CE weights -> Focal standard -> Focal weights
    """
    def __init__(self, alpha=None, gamma=0.0, reduction='mean'):
        super(FocalLossMulticlass, self).__init__()
        
        # alpha: classes weights
        self.alpha = alpha 
        
        # gamma: ???
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculates CE standard or with weights
        # reduction=none gives each separate pixel error
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')

        # get right class probs
        pt = torch.exp(-ce_loss)

        # aplly focal loss (1 - pt)^gamma * CE
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # mean of every pixel in the image
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



def mosaic2x2(dataloader):
    """
    create mosaic 2x2 (256x256) with unique ids
    """
    # get image and mask from another batch in loader
    images, masks = next(iter(dataloader))

    # separates 4 first images and their instance masks
    img1, img2, img3, img4 = images[0], images[1], images[2], images[3]
    mask1, mask2, mask3, mask4 = masks[0], masks[1], masks[2], masks[3]
    
    # mask 1 top left: intact, get its biggest id
    max_id_1 = mask1.max().item()

    # mask 2 top right: add max_id_1 only where it isnt background
    mask2_offset = mask2.clone()
    mask2_offset[mask2 > 0] += max_id_1
    max_id_2 = mask2_offset.max().item() if mask2_offset.max().item() > 0 else max_id_1

    # mask 3 bottom left: add max_id_2
    mask3_offset = mask3.clone()
    mask3_offset[mask3 > 0] += max_id_2
    max_id_3 = mask3_offset.max().item() if mask3_offset.max().item() > 0 else max_id_2

    # mask 4 bottom right: add max_id_3
    # Máscara 4 (Fundo Direita): Adicionamos o max_id_3
    mask4_offset = mask4.clone()
    mask4_offset[mask4 > 0] += max_id_3
    

    # sewing images 
    top_img = torch.cat((img1, img2), dim=2)    
    bottom_img = torch.cat((img3, img4), dim=2)
    mosaic_img = torch.cat((top_img, bottom_img), dim=1) #  [3, 256, 256]

    # sewing corrected images
    top_mask = torch.cat((mask1, mask2_offset), dim=1)
    bottom_mask = torch.cat((mask3_offset, mask4_offset), dim=1)
    mosaic_mask = torch.cat((top_mask, bottom_mask), dim=0) #  [256, 256]

    # add back batch dim to pass to the net
    # [3, 256, 256] -> [1, 3, 256, 256]
    return mosaic_img.unsqueeze(0), mosaic_mask
