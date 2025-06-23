import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from picosam2_model_distillation import PicoSAM2Dataset, PicoSAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_SIZE = 96
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
COCO_IMG_ROOT = os.path.join(BASE_DIR, "..", "dataset", "val2017")
COCO_ANN_FILE = os.path.join(BASE_DIR, "..", "dataset", "annotations", "instances_val2017.json")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(SCRIPT_DIR, "images/mask_comparison_all_models.png")
sample_index = 30

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1)

def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    image = np.array(image).copy()
    overlay = image.copy()
    mask = mask.astype(bool)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    return overlay.astype(np.uint8)

def plot_comparison(image_tensor, gt_mask, preds_dict, prompt_coords, out_path="comparison.png"):
    img = unnormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    gt_mask = gt_mask.squeeze().cpu().numpy() > 0.5

    fig, axs = plt.subplots(1, len(preds_dict) + 2, figsize=(4 * (len(preds_dict) + 2), 5))
    
    axs[0].imshow(img)
    axs[0].scatter([prompt_coords[0]], [prompt_coords[1]], c='red', s=40)
    axs[0].set_title("Input", fontsize=16, weight='bold', pad=20)
    axs[0].axis('off')

    axs[1].imshow(overlay_mask(img, gt_mask))
    axs[1].set_title("Ground Truth", fontsize=16, weight='bold', pad=20)
    axs[1].axis('off')

    for i, (label, mask_tensor) in enumerate(preds_dict.items()):
        mask = (torch.sigmoid(mask_tensor.squeeze()) > 0.5).cpu().numpy()
        axs[i + 2].imshow(overlay_mask(img, mask))
        axs[i + 2].set_title(label, fontsize=16, pad=20)
        axs[i + 2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(out_path, dpi=300)
    plt.close()

dataset = PicoSAM2Dataset(COCO_IMG_ROOT, COCO_ANN_FILE, image_size=IMAGE_SIZE)
sample_img, gt_mask, prompt_coords, img_id = dataset[sample_index]
sample_img = sample_img.unsqueeze(0).to(DEVICE)

def load_picosam2(path):
    model = PicoSAM2().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, path)))
    return model.eval()

scratch = load_picosam2("PicoSAM2_epoch1.pt")
distilled = load_picosam2("PicoSAM2_student_epoch1.pt")
quant = load_picosam2("PicoSAM2_student_epoch1.pt")

with torch.no_grad():
    pred_scratch = scratch(sample_img).cpu()
    pred_distilled = distilled(sample_img).cpu()
    pred_quant = quant(sample_img).cpu()

def load_sam2(cfg, ckpt):
    return SAM2ImagePredictor(build_sam2(cfg, os.path.join(CKPT_DIR, ckpt), device=DEVICE, mode="eval"))

sam_models = {
    "SAM2.1 Large": load_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    "SAM2.1 Base+": load_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "SAM2.1 Small": load_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "SAM2.1 Tiny": load_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
}

unnorm_img = (unnormalize(sample_img[0].cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
pt = np.array([[[unnorm_img.shape[1]//2, unnorm_img.shape[0]//2]]])
lbl = np.array([[1]])

sam_preds = {}
for label, predictor in sam_models.items():
    predictor.set_image(unnorm_img)
    masks, scores, _ = predictor.predict(point_coords=pt, point_labels=lbl, return_logits=False)
    best_idx = np.argmax(scores)
    sam_preds[label] = torch.tensor(masks[best_idx]).unsqueeze(0).unsqueeze(0)

predictions = {
    "SAM2.1 Large": sam_preds["SAM2.1 Large"],
    "SAM2.1 Base+": sam_preds["SAM2.1 Base+"],
    "SAM2.1 Small": sam_preds["SAM2.1 Small"],
    "SAM2.1 Tiny": sam_preds["SAM2.1 Tiny"],
    "Supervised Model": pred_scratch,
    "PicoSAM2": pred_distilled,
    "Q-PicoSAM2": pred_quant,
}

plot_comparison(sample_img[0].cpu(), gt_mask, predictions, prompt_coords, out_path=relative_path)