import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2 # uv add opencv-python
from skimage.segmentation import watershed

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models


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



def plot_fracasso_densidade(model, val_loader, device):
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