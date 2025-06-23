import os
import matplotlib.pyplot as plt
from adjustText import adjust_text

results_latency = [
    {"name": "SAM-H", "size": 2420.3, "latency": 2392},
    {"name": "SAM-L", "size": 1174.9, "latency": 1146},
    {"name": "SAM-B", "size": 343.3, "latency": 368.8},
    {"name": "FastSAM", "size": 275.6, "latency": 153.6},
    {"name": "EfficientSAM-Ti", "size": 118.2, "latency": 81},
    {"name": "SlimSAM-77", "size": 51.0, "latency": 110},
    {"name": "MobileSAM", "size": 37.0, "latency": 38.4},
    {"name": "TinySAM", "size": 37.0, "latency": 38.4},
    {"name": "Q-TinySAM", "size": 16.0, "latency": 24},
    {"name": "PicoSAM2", "size": 4.87, "latency": 2.54},
]

colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:pink', 'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan', 'gold'
]

markers = [
    'o', 's', '^', 'D', 'P', 'X', 'd', 'v', '*', '<', '>'
]

def create_latency_vs_size_plot(data, x_key, x_label, y_label, title, filename):
    plt.figure(figsize=(9, 6))
    texts = []
    for i, r in enumerate(data):
        x_value = r[x_key]
        y_value = r["latency"]
        if x_value is None or y_value is None:
            continue
        plt.scatter(x_value, y_value, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                    s=80, edgecolor='black', label=r["name"])
        text = plt.text(
            x_value, y_value, r["name"],
            fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )
        texts.append(text)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label + " (log scale)", fontsize=14)
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

create_latency_vs_size_plot(
    results_latency,
    "size",
    "Model Size (MB)",
    "Latency (ms)",
    "Latency vs Model Size",
    "images/latency_vs_size_log.png"
)
