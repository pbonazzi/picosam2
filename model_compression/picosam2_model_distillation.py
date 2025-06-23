import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from pycocotools.coco import COCO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_SIZE = 96
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_ROOT = os.path.join(BASE_DIR, "../dataset/train2017")
ANN_FILE = os.path.join(BASE_DIR, "../dataset/annotations/instances_train2017.json")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "../checkpoints/sam2.1_hiera_tiny.pt")
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"
OUTPUT_DIR = os.path.join(BASE_DIR, "../checkpoints")
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


class PicoSAM2(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def depthwise_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder_stage1 = depthwise_conv(in_channels, 48)
        self.down1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage2 = depthwise_conv(48, 96)
        self.down2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage3 = depthwise_conv(96, 160)
        self.down3 = nn.Conv2d(160, 160, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_stage4 = depthwise_conv(160, 256)
        self.down4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)

        self.bottleneck = depthwise_conv(256, 320)

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(320, 192))
        self.skip_conn4 = nn.Conv2d(256, 192, kernel_size=1)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(192, 128))
        self.skip_conn3 = nn.Conv2d(160, 128, kernel_size=1)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(128, 80))
        self.skip_conn2 = nn.Conv2d(96, 80, kernel_size=1)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), depthwise_conv(80, 40))
        self.skip_conn1 = nn.Conv2d(48, 40, kernel_size=1)

        self.output_head = nn.Conv2d(40, 1, kernel_size=1)

    def forward(self, x):
        feat1 = self.encoder_stage1(x)
        feat2 = self.encoder_stage2(self.down1(feat1))
        feat3 = self.encoder_stage3(self.down2(feat2))
        feat4 = self.encoder_stage4(self.down3(feat3))
        bottleneck_out = self.bottleneck(self.down4(feat4))

        upsample1 = self.up1(bottleneck_out) + self.skip_conn4(feat4)
        upsample2 = self.up2(upsample1) + self.skip_conn3(feat3)
        upsample3 = self.up3(upsample2) + self.skip_conn2(feat2)
        upsample4 = self.up4(upsample3) + self.skip_conn1(feat1)

        return self.output_head(upsample4)


class PicoSAM2Dataset(Dataset):
    def __init__(self, image_root, annotation_file, image_size):
        self.coco = COCO(annotation_file)
        self.image_dir = image_root
        self.image_size = image_size
        self.image_ids = self.coco.getImgIds()
        self.annotations = [
            ann for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.image_ids))
            if "segmentation" in ann and ann.get("iscrowd", 0) == 0
        ]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        file_name = img_info.get('file_name', f"{img_info['id']:012d}.jpg")
        img_path = os.path.join(self.image_dir, file_name)

        image = Image.open(img_path).convert("RGB")
        mask = self.coco.annToMask(ann)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
        else:
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            mask_center_x = (min_x + max_x) // 2
            mask_center_y = (min_y + max_y) // 2

            if mask[mask_center_y, mask_center_x]:
                center_x, center_y = mask_center_x, mask_center_y
            else:
                distances = (xs - mask_center_x) ** 2 + (ys - mask_center_y) ** 2
                closest_idx = np.argmin(distances)
                center_x, center_y = xs[closest_idx], ys[closest_idx]



        left = max(0, center_x - self.image_size // 2)
        top = max(0, center_y - self.image_size // 2)
        right = min(image.size[0], left + self.image_size)
        bottom = min(image.size[1], top + self.image_size)

        cropped_img = image.crop((left, top, right, bottom)).resize((self.image_size, self.image_size), Image.BILINEAR)
        cropped_mask = Image.fromarray(mask[top:bottom, left:right]).resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = self.transform(cropped_img)
        mask_tensor = torch.tensor(np.array(cropped_mask), dtype=torch.float32).unsqueeze(0)
        prompt_coords = (center_x - left, center_y - top)
        image_id = ann["image_id"]

        return image_tensor, mask_tensor, prompt_coords, image_id


def bce_dice_loss(pred_mask, gt_mask):
    pred_mask = torch.sigmoid(pred_mask)

    bce = F.binary_cross_entropy(pred_mask, gt_mask)
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return bce + dice

def mse_dice_loss(pred_mask, soft_mask):
    pred_mask = torch.sigmoid(pred_mask)
    soft_mask = torch.sigmoid(soft_mask)

    mse = F.mse_loss(pred_mask, soft_mask)
    intersection = (pred_mask * soft_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + soft_mask.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()
    return mse + dice


def compute_iou(pred_mask, target_mask):
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()
    target_binary = (torch.sigmoid(target_mask) > 0.5).cpu().numpy()

    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return intersection / union if union > 0 else 1.0


def save_visualization(image_tensor, pred_mask, gt_mask, step):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 0.229 + 0.485).clip(0, 1)
    pred = torch.sigmoid(pred_mask[0]).squeeze().detach().cpu().numpy()
    gt = torch.sigmoid(gt_mask[0]).squeeze().detach().cpu().numpy()

    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].scatter([cx], [cy], c="red", s=10)
    axes[0].set_title("Image + Prompt")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Student Prediction")

    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("Teacher Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(IMAGE_OUTPUT_DIR, f"vis_step{step}.png")
    plt.savefig(save_path)
    plt.close()
    wandb.log({"visualization": wandb.Image(save_path)})


def train():
    wandb.init(project="PicoSAM2-distillation", config={"img_size": IMAGE_SIZE, "epochs": NUM_EPOCHS})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=device, mode="eval")
    teacher_predictor = SAM2ImagePredictor(teacher_model)

    student_model = PicoSAM2().to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 1000))

    dataset = PicoSAM2Dataset(IMG_ROOT, ANN_FILE, IMAGE_SIZE)
    val_size = max(1, len(dataset) // 20)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    vis_interval = max(1, len(train_loader) // 10)

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        total_loss, total_iou, num_samples = 0, 0, 0

        for batch_idx, (images, gt_masks, prompts, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            prompts = torch.stack([
                p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.int64)
                for p in prompts
            ]).to(device)

            if prompts.shape[0] == 2 and prompts.shape[1] == images.size(0):
                prompts = prompts.transpose(0, 1)

            pred_masks = student_model(images)

            teacher_masks = []
            selected_scores = []
            for i in range(images.size(0)):
                img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 0.229 + 0.485) * 255
                img_np = img_np.clip(0, 255).astype(np.uint8)
                teacher_predictor.set_image(img_np)

                coords = np.array([[[int(prompts[i][0]), int(prompts[i][1])]]])
                labels = np.array([[1]])
                _, scores, mask_logits = teacher_predictor.predict(
                    point_coords=coords, point_labels=labels, return_logits=True
                )
                best_idx = np.argmax(scores)
                mask_tensor = torch.tensor(mask_logits[best_idx]).unsqueeze(0).unsqueeze(0)
                mask_tensor = F.interpolate(mask_tensor, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
                teacher_masks.append(mask_tensor)
                selected_scores.append(scores[best_idx])

            teacher_masks = torch.cat(teacher_masks).to(device)


            confidence = torch.tensor(selected_scores, device=device)
            loss_teacher = mse_dice_loss(pred_masks, teacher_masks)
            loss_gt = bce_dice_loss(pred_masks, gt_masks)

            alpha = confidence
            loss = (alpha * loss_teacher + (1 - alpha) * loss_gt).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_iou = compute_iou(pred_masks, gt_masks)
            total_loss += loss.item() * images.size(0)
            total_iou += batch_iou * images.size(0)
            num_samples += images.size(0)

            wandb.log({
                "batch_loss": loss.item(),
                "loss_teacher": loss_teacher.item(),
                "teacher confidence": confidence.mean().item(),
                "loss_gt": loss_gt.item(),
                "batch_mIoU": batch_iou,
                "epoch": epoch + 1
            })


            if batch_idx % vis_interval == 0:
                save_visualization(images[0], pred_masks[0:1], teacher_masks[0:1],
                                   step=epoch * len(train_loader) + batch_idx)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / num_samples,
            "train_mIoU": total_iou / num_samples
        })

        student_model.eval()
        val_loss, val_iou, val_samples = 0, 0, 0

        with torch.no_grad():
            for images, gt_masks, prompts, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Val"):
                images = images.to(device)
                gt_masks = gt_masks.to(device)
                prompts = torch.stack([
                    p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.int64)
                    for p in prompts
                ]).to(device)

                if prompts.shape[0] == 2 and prompts.shape[1] == images.size(0):
                    prompts = prompts.transpose(0, 1)

                pred_masks = student_model(images)

                teacher_masks = []
                for i in range(images.size(0)):
                    img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 0.229 + 0.485) * 255
                    img_np = img_np.clip(0, 255).astype(np.uint8)
                    teacher_predictor.set_image(img_np)

                    coords = np.array([[[int(prompts[i][0]), int(prompts[i][1])]]])
                    labels = np.array([[1]])
                    _, scores, mask_logits = teacher_predictor.predict(
                        point_coords=coords, point_labels=labels, return_logits=True
                    )
                    best_idx = np.argmax(scores)
                    mask_tensor = torch.tensor(mask_logits[best_idx]).unsqueeze(0).unsqueeze(0)
                    mask_tensor = F.interpolate(mask_tensor, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
                    teacher_masks.append(mask_tensor)

                teacher_masks = torch.cat(teacher_masks).to(device)
                
                alpha = 0.5
                loss_teacher = mse_dice_loss(pred_masks, teacher_masks)
                loss_gt = bce_dice_loss(pred_masks, gt_masks)


                loss = alpha * loss_teacher + (1 - alpha) * loss_gt

                val_loss += loss.item() * images.size(0)
                val_iou += compute_iou(pred_masks, gt_masks) * images.size(0)
                val_samples += images.size(0)

        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss / val_samples,
            "val_mIoU": val_iou / val_samples
        })

        save_path = os.path.join(OUTPUT_DIR, f"PicoSAM2_student_epoch{epoch + 1}.pt")
        torch.save(student_model.state_dict(), save_path)



if __name__ == "__main__":
    train()
