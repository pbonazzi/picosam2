import os
import matplotlib.pyplot as plt
import numpy as np

results = [
    {"name": "SAM-H", "size": 2420.3},
    {"name": "SAM-L", "size": 1174.9},
    {"name": "SAM-B", "size": 343.3},
    {"name": "EfficientViT-L0-SAM", "size": 118.2},
    {"name": "MobileSAM", "size": 37.0},
    {"name": "EdgeSAM", "size": 37.0},
    {"name": "LiteSAM", "size": 16.0},
    {"name": "SAM2.1 Large", "size": 857.0},
    {"name": "SAM2.1 Base Plus", "size": 309.0},
    {"name": "SAM2.1 Small", "size": 176.0},
    {"name": "SAM2.1 Tiny", "size": 149.0},
    {"name": "PicoSAM2", "size": 4.87},
    {"name": "Q-PicoSAM2", "size": 1.22},
]

colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink',
    'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan', 'gold', 'darkblue',
    'lightcoral', 'limegreen', 'peru', 'slateblue', 'darkslategray'
]

def plot_blob_chart(results, title, filename):
    x_positions = np.arange(len(results))
    sizes = [r['size'] for r in results]
    max_size = max(sizes)
    area_scaled = [(s / max_size) * 3000 for s in sizes]

    plt.figure(figsize=(12, 5))
    for i, (r, area, color) in enumerate(zip(results, area_scaled, colors)):
        plt.scatter(x_positions[i], 0, s=area, color=color, alpha=0.8, edgecolor='black')
        plt.text(x_positions[i], 0.5, r["name"], ha='center', va='bottom', fontsize=9, rotation=40)
        plt.text(x_positions[i], -0.3, f"{r['size']:.1f} MB", ha='center', va='top', fontsize=8)

    plt.xticks([])
    plt.yticks([])
    plt.ylim(-0.7, 0.9)
    plt.xlim(-1, len(results))
    plt.title(title, fontsize=14)
    plt.grid(False)
    plt.box(False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "images", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

plot_blob_chart(
    sorted(results, key=lambda x: -x["size"]),
    "Model Size Comparison",
    "model_size_blob_chart_sorted.png"
)

tiny_index = next(i for i, r in enumerate(results) if r["name"] == "SAM2.1 Tiny")
results_from_tiny = [r for r in results if r["size"] <= results[tiny_index]["size"]]
plot_blob_chart(
    sorted(results_from_tiny, key=lambda x: -x["size"]),
    "Model Size Comparison",
    "model_size_blob_chart_compact.png"
)
