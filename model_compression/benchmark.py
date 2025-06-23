import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
import model_compression_toolkit as mct
from PIL import Image

from picosam2_model_distillation import PicoSAM2Dataset, PicoSAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_SIZE = 96
NUM_SAMPLES = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
COCO_IMG_ROOT = os.path.join(BASE_DIR, "..", "dataset", "val2017")
COCO_ANN_FILE = os.path.join(BASE_DIR, "..", "dataset", "annotations", "instances_val2017.json")
LVIS_IMG_ROOT = os.path.join(BASE_DIR, "..", "dataset", "val2017_lvis")
LVIS_ANN_FILE = os.path.join(BASE_DIR, "..", "dataset", "annotations", "lvis_v1_val.json")

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0,1)

def calc_miou(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).cpu().numpy()
    targets = (targets > 0.5).cpu().numpy()
    return np.mean([
        (np.logical_and(p,t).sum() / np.logical_or(p,t).sum()) if np.logical_or(p,t).sum() else 1.0
        for p,t in zip(preds, targets)
    ])

def calc_map_iou_range(preds, targets, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    preds = torch.sigmoid(preds).cpu().numpy()
    targets = targets.cpu().numpy()
    aps = []

    for iou_thresh in iou_thresholds:
        ap_per_thresh = []
        for p, t in zip(preds, targets):
            if t.sum() == 0: continue
            p_bin = p > 0.5
            t_bin = t > 0.5
            iou = np.logical_and(p_bin, t_bin).sum() / (np.logical_or(p_bin, t_bin).sum() + 1e-8)
            ap_per_thresh.append(1.0 if iou >= iou_thresh else 0.0)
        if ap_per_thresh:
            aps.append(np.mean(ap_per_thresh))

    return float(np.mean(aps)) if aps else 0.0

def evaluate_picosam(model, loader, name):
    print(f"\nEvaluating: {name}")
    preds, gts, mious = [], [], []
    for i, (x, y, _, _) in enumerate(tqdm(loader)):
        if i >= NUM_SAMPLES: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            pred = model(x)
        pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred)
        gts.append(y)
        mious.append(calc_miou(pred, y))
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    print(f"{name} -> mIoU: {np.mean(mious):.4f}, mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}")

def evaluate_sam2(predictor, dataset, name):
    print(f"\nEvaluating: {name}")
    preds, gts, mious = [], [], []
    for i in range(min(NUM_SAMPLES, len(dataset))):
        image, mask, prompt_coords, _ = dataset[i]
        image_np = (unnormalize(image).permute(1,2,0).numpy() * 255).astype(np.uint8)
        predictor.set_image(image_np)
        pt = np.array([[prompt_coords]])
        lbl = np.array([[1]])
        masks, scores, _ = predictor.predict(point_coords=pt, point_labels=lbl, return_logits=False)
        best_idx = np.argmax(scores)
        pred = torch.tensor(masks[best_idx]).unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        preds.append(pred)
        gts.append(mask.unsqueeze(0))
        mious.append(calc_miou(pred, mask.unsqueeze(0)))
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    print(f"{name} -> mIoU: {np.mean(mious):.4f}, mAP@[0.5:0.95]: {calc_map_iou_range(preds, gts):.4f}")

if __name__ == "__main__":
    coco_data = PicoSAM2Dataset(COCO_IMG_ROOT, COCO_ANN_FILE, image_size=IMAGE_SIZE)
    coco_loader = DataLoader(coco_data, batch_size=1, shuffle=False)
    lvis_data = PicoSAM2Dataset(LVIS_IMG_ROOT, LVIS_ANN_FILE, image_size=IMAGE_SIZE)
    lvis_loader = DataLoader(lvis_data, batch_size=1, shuffle=False)

    scratch = PicoSAM2().to(DEVICE)
    scratch.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_epoch1.pt"))); scratch.eval()
    distilled = PicoSAM2().to(DEVICE)
    distilled.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_student_epoch1.pt"))); distilled.eval()
    quant = PicoSAM2().to("cpu")
    quant.load_state_dict(torch.load(os.path.join(CKPT_DIR, "PicoSAM2_student_epoch1.pt"))); quant.eval()

    def repr_dataset():
        val_iter = itertools.cycle(coco_loader)
        def generator():
            for _ in range(10):
                yield [next(val_iter)[0].cpu()]
        return generator

    tpc = mct.get_target_platform_capabilities("pytorch", "imx500")
    quantized, _ = mct.ptq.pytorch_post_training_quantization(
        quant,
        representative_data_gen=repr_dataset(), 
        target_platform_capabilities=tpc
    )


    sam_variants = {
        "SAM2.1 Large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        "SAM2.1 Base+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "SAM2.1 Small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "SAM2.1 Tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt")
    }
    sam_predictors = {name: SAM2ImagePredictor(build_sam2(cfg, os.path.join(CKPT_DIR, ckpt), device=DEVICE, mode="eval")) for name, (cfg, ckpt) in sam_variants.items()}

    evaluate_picosam(scratch, coco_loader, "PicoSAM2 Trained (COCO)")
    evaluate_picosam(distilled, coco_loader, "PicoSAM2 Distilled (COCO)")
    evaluate_picosam(quantized, coco_loader, "PicoSAM2 Quantized (COCO)")
    evaluate_picosam(scratch, lvis_loader, "PicoSAM2 Trained (LVIS)")
    evaluate_picosam(distilled, lvis_loader, "PicoSAM2 Distilled (LVIS)")
    evaluate_picosam(quantized, lvis_loader, "PicoSAM2 Quantized (LVIS)")

    for name, predictor in sam_predictors.items():
        evaluate_sam2(predictor, coco_data, f"{name} (COCO)")
        evaluate_sam2(predictor, lvis_data, f"{name} (LVIS)")
