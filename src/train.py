import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from .utils import extract_instances_watershed
from .metrics import compute_instance_metrics

def train_model_track_a(
    model,
    optimizer: optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    model_name: str  
):
    """
    Trains the model after the track A.

    model (nn.Module): The instantiate model to train.
    optimizer (optim.Optimizer): Optimizer to use.
    criterion (nn.modules.loss._Loss): Criterion or loss function to use.
    train_loader (DataLoader): Train dataset loader.
    val_loader (DataLoader): Validation dataset loader.
    device (Literal['cpu', 'cuda']): Device to use.
    epochs (int): Number of epochs to train.
    model_name (str): Name of the model to print and save weights file.
    """
    
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for images, masks_semantic, masks_instance_gt in train_loader:
            images = images.to(device)
            masks_semantic = masks_semantic.to(device) #  0, 1 or 2

            optimizer.zero_grad()
            logits = model(images)

            # CrossEntropyLoss compares output [B, 3, H, W] with answer [B, H, W]
            loss = criterion(logits, masks_semantic)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        loss_media_treino = train_loss / len(train_loader)
        print(f"\nÉpoca {epoch+1}/{epochs} | Loss Treino: {loss_media_treino:.4f}")


        model.eval()

        mAPs_inst = []
        erros_contagem_inst = []

        with torch.no_grad():
            for images, masks_semantic, masks_instance_gt in val_loader:
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

        print(f"  [{model_name}] mAP: {map_final:.4f} | Erro Abs. Contagem: {erro_final:.4f}")

    # saves winning model
    torch.save(model.state_dict(), f'../{model_name}.pth')
    print("\nModelo salvo com sucesso!")