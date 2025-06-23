import os
import matplotlib.pyplot as plt
from adjustText import adjust_text

results = [
    {"name": "SAM-H", "coco_miou": 0.536, "lvis_miou": 0.605, "size": 2420.3},
    {"name": "SAM-L", "coco_miou": 0.536, "lvis_miou": 0.605, "size": 1174.9},
    {"name": "SAM-B", "coco_miou": 0.536, "lvis_miou": 0.605, "size": 343.3},
    {"name": "MobileSAM", "coco_miou": 0.509, "lvis_miou": 0.521, "size": 37.0},
    {"name": "EdgeSAM", "coco_miou": 0.480, "lvis_miou": 0.537, "size": 37.0},
    {"name": "SAM2.1 Large", "coco_miou": 0.5217, "lvis_miou": 0.5625, "size": 857.0},
    {"name": "SAM2.1 Base Plus", "coco_miou": 0.5327, "lvis_miou": 0.5474, "size": 309.0},
    {"name": "SAM2.1 Small", "coco_miou": 0.5220, "lvis_miou": 0.5435, "size": 176.0},
    {"name": "SAM2.1 Tiny", "coco_miou": 0.5045, "lvis_miou": 0.5268, "size": 149.0},
    {"name": "Supervised Model", "coco_miou": 0.5300, "lvis_miou": 0.4140, "size": 4.87},
    {"name": "PicoSAM2", "coco_miou": 0.5193, "lvis_miou": 0.4488, "size": 4.87},
    {"name": "Q-PicoSAM2", "coco_miou": 0.5047, "lvis_miou": 0.4509, "size": 1.22},
]


colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
    'tab:purple', 'tab:pink', 'tab:brown', 'tab:gray',
    'tab:olive', 'tab:cyan', 'gold', 'darkblue'
]

markers = [
    'o', 's', '^', 'D', 'P', 'X', 'd', 'v',
    '*', '<', '>', 'H'
]

def create_plot(results, x_key, x_label, y_label, title, filename):
    plt.figure(figsize=(9, 6))
    texts = []
    for i, r in enumerate(results):
        x_value = r[x_key]
        y_value = r["size"]
        if x_value is None:
            continue
        plt.scatter(x_value, y_value, color=colors[i], marker=markers[i], s=80, edgecolor='black', label=r["name"])
        text = plt.text(
            x_value, y_value, r["name"],
            fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
        texts.append(text)

    plt.yscale('log')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label + " (log scale)", fontsize=14)
    plt.grid(True, which='both', ls="--", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

create_plot(results, "coco_miou", "COCO mIoU", "Model Size (MB)", "Model Size vs COCO mIoU", "images/coco_miou_vs_size_log_others.png")
create_plot(results, "lvis_miou", "LVIS mIoU", "Model Size (MB)", "Model Size vs LVIS mIoU", "images/lvis_miou_vs_size_log_others.png")

