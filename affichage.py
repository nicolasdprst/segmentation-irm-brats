import matplotlib.pyplot as plt
import numpy as np
import torch


def save_comparison_plot(model, val_loader, device, filename="brats_results.png", model_name="SegFormer + LoRA"):
    model.eval()
    images, labels = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    img_np = images[0].cpu().numpy()
    lbl_np = labels[0].cpu().numpy()
    prd_np = preds[0].cpu().numpy()

    # On masque le fond (classe 0) pour ne colorer que les tissus tumoraux sur l'IRM
    lbl_masked = np.ma.masked_where(lbl_np == 0, lbl_np)
    prd_masked = np.ma.masked_where(prd_np == 0, prd_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Canal T1ce (index 1) : séquence la plus parlante avec produit de contraste
    axes[0].imshow(img_np[1], cmap="gray")
    axes[0].set_title("IRM (T1ce)")
    axes[0].axis("off")

    axes[1].imshow(img_np[1], cmap="gray")
    axes[1].imshow(lbl_masked, alpha=0.5, cmap="jet")
    axes[1].set_title("Vérité terrain")
    axes[1].axis("off")

    axes[2].imshow(img_np[1], cmap="gray")
    axes[2].imshow(prd_masked, alpha=0.5, cmap="jet")
    axes[2].set_title(f"Prédiction ({model_name})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {filename}")