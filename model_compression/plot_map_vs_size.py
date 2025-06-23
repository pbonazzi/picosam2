import os
import matplotlib.pyplot as plt
from adjustText import adjust_text

results = [
    {"name": "SAM2.1 Large", "coco_miou": 0.5217, "lvis_miou": 0.5625, "coco_map": 0.3569, "lvis_map": 0.4240, "size": 857.0},
    {"name": "SAM2.1 Base Plus", "coco_miou": 0.5327, "lvis_miou": 0.5474, "coco_map": 0.33683, "lvis_map": 0.3958, "size": 309.0},
    {"name": "SAM2.1 Small", "coco_miou": 0.5220, "lvis_miou": 0.5435, "coco_map": 0.3489, "lvis_map": 0.3860, "size": 176.0},
    {"name": "SAM2.1 Tiny", "coco_miou": 0.5045, "lvis_miou": 0.5268, "coco_map": 0.3316, "lvis_map": 0.3708, "size": 149.0},
    {"name": "Supervised Model", "coco_miou": 0.5300, "lvis_miou": 0.4140, "coco_map": 0.3067, "lvis_map": 0.2038, "size": 4.87},
    {"name": "PicoSAM2", "coco_miou": 0.5193, "lvis_miou": 0.4488, "coco_map": 0.2996, "lvis_map": 0.2546, "size": 4.87},
    {"name": "Q-PicoSAM2", "coco_miou": 0.5047, "lvis_miou": 0.4509, "coco_map": 0.2806, "lvis_map": 0.2577, "size": 1.22},
]


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:red', 'tab:olive', 'tab:cyan']
markers = ['o', 's', '^', 'D', 'P', 'X', 'd', 'v']

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

create_plot(results, "coco_map", "COCO mAP@[0.5:0.95]", "Model Size (MB)", "Model Size vs COCO mAP", "images/coco_map_vs_size_log.png")
create_plot(results, "lvis_map", "LVIS mAP@[0.5:0.95]", "Model Size (MB)", "Model Size vs LVIS mAP", "images/lvis_map_vs_size_log.png")
