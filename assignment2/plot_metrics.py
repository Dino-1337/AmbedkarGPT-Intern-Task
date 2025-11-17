import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the summary results
with open("test_results_real.json", "r") as f:
    data = json.load(f)["summary"]

os.makedirs("plots", exist_ok=True)

# Extract metrics
labels = ["small", "medium", "large"]
precision = [data[k]["precision_at_3"] for k in labels]
faithfulness = [data[k]["mean_faithfulness"] for k in labels]
cosine = [data[k]["mean_cosine"] for k in labels]
hit_rate = [data[k]["hit_rate"] for k in labels]
mrr = [data[k]["mrr"] for k in labels]

# Helper to create each plot
def plot_bar(values, title, ylabel, filename):
    x = np.arange(len(labels))
    plt.figure(figsize=(6,4))
    bars = plt.bar(x, values)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}",
                 ha='center', va='bottom', fontsize=8)

    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max(values) + 0.1)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=200)
    plt.close()

# Generate all plots
plot_bar(precision, "Precision@3 Across Chunk Sizes", "Precision@3", "precision")
plot_bar(faithfulness, "Faithfulness Across Chunk Sizes", "Faithfulness", "faithfulness")
plot_bar(cosine, "Cosine Similarity Across Chunk Sizes", "Cosine Similarity", "cosine")
plot_bar(hit_rate, "Hit Rate Across Chunk Sizes", "Hit Rate", "hit_rate")
plot_bar(mrr, "MRR Across Chunk Sizes", "Mean Reciprocal Rank", "mrr")

print("All plots saved to assignment2/plots/")
