import argparse
import gc
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from monai.apps import DecathlonDataset
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
)
from peft import LoraConfig, get_peft_model
from transformers import SegformerForSemanticSegmentation

from affichage import save_comparison_plot


class BraTS2DSliceDataset(Dataset):
    def __init__(self, monai_dataset, max_patients=100, slices_per_patient=8):
        self.samples = []
        n_patients = min(len(monai_dataset), max_patients)
        print(f"Extraction des coupes 2D pour {n_patients} patients...")

        for i in range(n_patients):
            item = monai_dataset[i]
            vol_img = item["image"]
            vol_lbl = item["label"]

            # Sélection des coupes axiales où la tumeur est présente (> 150 voxels)
            tumor_pixels = (vol_lbl[0] > 0).sum(dim=(0, 1))
            valid_z = torch.where(tumor_pixels > 150)[0]

            if len(valid_z) > 0:
                indices = torch.linspace(
                    0, len(valid_z) - 1, steps=min(slices_per_patient, len(valid_z))
                ).long()
                for z in valid_z[indices]:
                    # On stocke en float16 et uint8 pour réduire drastiquement l'empreinte RAM
                    img_slice = vol_img[[0, 2, 3], :, :, z].clone().half()
                    lbl_slice = vol_lbl[0, :, :, z].clone().to(torch.uint8)
                    self.samples.append((img_slice, lbl_slice))

            # Libération immédiate du volume 3D complet
            del item, vol_img, vol_lbl
            gc.collect()

        print(f"Dataset 2D prêt : {len(self.samples)} coupes extraites.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, lbl = self.samples[idx]
        return img.float(), lbl.long()


def get_brats_dataloaders(root_dir="./data", train_patients=100, val_patients=15, batch_size=8):
    os.makedirs(root_dir, exist_ok=True)
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 140]),
    ])

    raw_train = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=transforms,
        download=True,
        cache_rate=0.0,
        num_workers=0,
    )
    raw_val = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=transforms,
        download=True,
        cache_rate=0.0,
        num_workers=0,
    )

    train_ds = BraTS2DSliceDataset(raw_train, max_patients=train_patients, slices_per_patient=8)
    val_ds = BraTS2DSliceDataset(raw_val, max_patients=val_patients, slices_per_patient=8)

    del raw_train, raw_val
    gc.collect()

    # Données déjà en mémoire : num_workers=0 évite les duplications mémoire et les blocages IPC
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def build_attention_unet(num_classes=4):
    return AttentionUnet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
    )


class LoRAVisionTransformerSeg(nn.Module):
    def __init__(self, num_classes=4, lora_r=8):
        super().__init__()
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)

        # On dégèle la tête de segmentation pour nos 4 classes
        for name, param in self.model.named_parameters():
            if "decode_head" in name:
                param.requires_grad = True

        self.model.print_trainable_parameters()

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # La tête SegFormer produit des logits à 1/4 de la résolution d'entrée
        return nn.functional.interpolate(
            outputs.logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def evaluate_model(model, val_loader, device, num_classes=4):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95.0, reduction="mean_batch"
    )

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds_oh = nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
            labels_oh = nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2)

            dice_metric(y_pred=preds_oh, y=labels_oh)
            try:
                hd95_metric(y_pred=preds_oh, y=labels_oh)
            except Exception:
                pass

    dice = dice_metric.aggregate().cpu().numpy()
    try:
        hd95 = hd95_metric.aggregate().cpu().numpy()
    except Exception:
        hd95 = np.full(num_classes - 1, np.nan)

    dice_metric.reset()
    hd95_metric.reset()

    return dice, hd95


def run_experiment(model, name, train_loader, val_loader, device, epochs=25, lr=3e-4):
    print(f"\n--- Entraînement : {name} ---")
    model = model.to(device)

    # Perte hybride adaptée au fort déséquilibre (fond > 95% des pixels)
    criterion = DiceFocalLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=0.6,
        lambda_focal=0.4,
        gamma=2.0,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dice, hd95 = evaluate_model(model, val_loader, device)

        print(
            f"[{name}] Époque {epoch:02d}/{epochs:02d} | Perte : {avg_loss:.4f} | "
            f"Dice moyen : {np.nanmean(dice) * 100:.1f}% "
            f"(œdème : {dice[0] * 100:.1f}%, nécrose : {dice[1] * 100:.1f}%, active : {dice[2] * 100:.1f}%) | "
            f"HD95 moyen : {np.nanmean(hd95):.2f} px"
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation IRM BraTS (LoRA vs Attention U-Net)")
    parser.add_argument(
        "--model",
        choices=["lora", "unet", "both"],
        default="both",
        help="Modèle à entraîner : 'lora', 'unet' ou 'both' pour comparer les deux",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Nombre d'époques par modèle")
    parser.add_argument("--batch-size", type=int, default=8, help="Taille des batchs")
    parser.add_argument("--train-patients", type=int, default=12, help="Nombre de patients pour l'entraînement")
    parser.add_argument("--val-patients", type=int, default=8, help="Nombre de patients pour la validation")
    parser.add_argument("--data-dir", default="./data", help="Dossier de cache des données")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique utilisé : {device}")

    train_loader, val_loader = get_brats_dataloaders(
        root_dir=args.data_dir,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
        batch_size=args.batch_size,
    )

    if args.model in ("lora", "both"):
        lora_model = run_experiment(
            LoRAVisionTransformerSeg(num_classes=4, lora_r=8),
            "SegFormer-B2 + LoRA",
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
        )
        save_comparison_plot(
            lora_model,
            val_loader,
            device,
            filename="resultats_lora.png",
            model_name="SegFormer-B2 + LoRA",
        )

    if args.model in ("unet", "both"):
        unet_model = run_experiment(
            build_attention_unet(num_classes=4),
            "Attention U-Net",
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
        )
        save_comparison_plot(
            unet_model,
            val_loader,
            device,
            filename="resultats_unet.png",
            model_name="Attention U-Net",
        )