"""Modulo of util functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.segmentation import watershed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
        for images, _, masks_gt in val_loader:
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


def tiles_inference_with_overlap(model, huge_img, cutting_points_x, cutting_points_y, tile_size=128):
    """
    inference in tiles with overlap on borders, meaning objects are sliced
    """
    model.eval()
    H, W = huge_img.shape[2], huge_img.shape[3]
    
    # empty canvas to glue results
    final_mask = np.zeros((H, W), dtype=np.int32)
    max_current_id = 0
    
    for y in cutting_points_y:
        for x in cutting_points_x:
            # cut tile_sizextile_size patch
            patch = huge_img[:, :, y:y+tile_size, x:x+tile_size]

            # go through net applying watershed
            with torch.no_grad():
                logits = model(patch)
            patch_insts = extract_instances_watershed(logits[0])

            # moves ids to avoid collapse with ellipses of previous patches
            patch_insts_offset = patch_insts.copy()
            foreground = patch_insts > 0
            patch_insts_offset[foreground] += max_current_id
            
            if patch_insts.max() > 0:
                max_current_id += patch_insts.max()

            # glue overwriting whats already there
            final_mask[y:y+tile_size, x:x+tile_size][foreground] = patch_insts_offset[foreground]
            
    return final_mask


def tiles_inference_corrected(model, huge_img, cuttting_points_x, cutting_points_y, tile_size=128):
    """
    Roda a inferência em tiles e aplica a correção lógica de fusão na borda.
    """
    model.eval()
    H, W = huge_img.shape[2], huge_img.shape[3]
    final_corrected_mask = np.zeros((H, W), dtype=np.int32)
    max_current_id = 0
    
    for y in cutting_points_y:
        for x in cuttting_points_x:
            patch = huge_img[:, :, y:y+tile_size, x:x+tile_size]
            
            with torch.no_grad():
                logits = model(patch)
            patch_insts = extract_instances_watershed(logits[0])
            
            patch_insts_offset = patch_insts.copy()
            foreground = patch_insts > 0
            patch_insts_offset[foreground] += max_current_id

            # math fusion in overlap area
            # look into area of global mosaic where patch will be glued
            overlap_area = final_corrected_mask[y:y+tile_size, x:x+tile_size]

            # get ellipses ids that net just found
            ids_new = np.unique(patch_insts_offset[foreground])
            
            for id_n in ids_new:
                # where is this new ellipse located?
                pixels_new_id = (patch_insts_offset == id_n)

                # which old ids are there just under it in mosaic?
                ids_under_old = np.unique(overlap_area[pixels_new_id])
                ids_under_old = ids_under_old[ids_under_old > 0] # Ignores bg
                
                for id_a in ids_under_old:
                    pixels_id_old = (overlap_area == id_a)

                    # intersection level between two ellipse pieces
                    intersection = np.logical_and(pixels_new_id, pixels_id_old).sum()
                    area_new = pixels_new_id.sum()

                    # if they overlap in more than 10% of new area, consider them the same object
                    if intersection / area_new > 0.10: 
                        # merge instances: rename id new with id old
                        patch_insts_offset[pixels_new_id] = id_a
                        break # pair found, go to the next ellipse
            
            # updates ids countes
            if patch_insts_offset.max() > max_current_id:
                max_current_id = patch_insts_offset.max()

            #glues the newly corrected ids to mosaic
            final_corrected_mask[y:y+tile_size, x:x+tile_size][foreground] = patch_insts_offset[foreground]
            
    return final_corrected_mask



def corrupt_image(images, type, intensity):
    """
    apply corruption directly in the batch image tensor, normalized in interval [0,1]
    """
    if intensity == 0:
        return images # 0 is the baseline
        
    if type == 'blur':
        # unfocus levels: kernel and sigma go up with intensity
        sigmas = [1.0, 2.0, 4.0]
        kernels = [3, 5, 9] 
        idx = intensity - 1
        # apply gaussian blur
        return TF.gaussian_blur(images, kernel_size=[kernels[idx], kernels[idx]], sigma=[sigmas[idx], sigmas[idx]])
        
    elif type == 'noise':
        # gaussian noise: increase std (granularity)
        stds = [0.05, 0.15, 0.30]
        noise = torch.randn_like(images) * stds[intensity - 1]
        # clamp clips pixel into valid image interval [0,1]
        return torch.clamp(images + noise, 0.0, 1.0)
        
    elif type == 'contrast':
        # reduces contrast by turning image more gray
        factors = [0.5, 0.2, 0.05]
        return TF.adjust_contrast(images, factors[intensity - 1])
        
    return images


def stress_test(model, val_loader, device):
    """
    run evaluation through all kinds of image corruption and plot mAP degradation curve
    """
    model.eval()
    
    corrup_types = ['blur', 'noise', 'contrast']
    intensities = [0, 1, 2, 3] # 0 = original, 1 = slight, 2 = medium, 3 = strong
    
    # map degradation dict
    mAP_curves = {type: [] for type in corrup_types}
    
    print("Iniciando teste de estresse (Parte 6)")
    
    with torch.no_grad():
        for type in corrup_types:
            print(f"\nAvaliando degradação por: {type.upper()}")
            
            for intensity in intensities:
                maps_current_intensity = []

                # use trilha A dataloader that gives back images with answers
                for images, masks_semantic, masks_instance_gt in val_loader:
                    images = images.to(device)
                    masks_gt_np = masks_instance_gt.cpu().numpy()

                    # apply math corruption to image batch
                    images_corrupted = corrupt_image(images, type, intensity)

                    # run trilha A net in damaged image
                    logits = model(images_corrupted)
                    
                    for i in range(images.size(0)):
                        # post processing with watershed using net output
                        pred_instance_mask = extract_instances_watershed(logits[i])
                        real_instance_mask = masks_gt_np[i]

                        # get real metrics
                        mAP_img, _ = compute_instance_metrics(pred_instance_mask, real_instance_mask)
                        maps_current_intensity.append(mAP_img)

                # final mean of current intensity to the graphic
                map_mean = np.mean(maps_current_intensity)
                mAP_curves[type].append(map_mean)
                print(f"  Nível {intensity} -> mAP: {map_mean:.4f}")
                
    # PLOT
    plt.figure(figsize=(10, 6))
    
    cores = {'blur': 'blue', 'noise': 'red', 'contrast': 'green'}
    labels = {'blur': 'Desfoque (Blur)', 'noise': 'Ruído Gaussiano', 'contrast': 'Baixo contrast'}
    
    for type in corrup_types:
        plt.plot(intensities, mAP_curves[type], marker='o', color=cores[type], label=labels[type], linewidth=2.5)
        
    plt.title("Parte 6: Curva de Degradação do mAP sob Corrupções", fontsize=14, fontweight='bold')
    plt.xlabel("intensity da Corrupção (0 = Imagem Original)", fontsize=12)
    plt.ylabel("Precisão Média de Instâncias (mAP)", fontsize=12)
    plt.xticks(intensities)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.show()
