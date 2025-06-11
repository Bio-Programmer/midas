# /content/drive/My Drive/midas/data/metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_named_class_distribution(df, index_to_label, title="Class distribution"):
    print(f"\n{title}:")
    proportions = df['label'].value_counts(normalize=True).sort_index()
    for idx, prop in proportions.items():
        label_name = index_to_label.get(idx, f"Unknown({idx})")
        print(f"{label_name:<5} ({idx}): {prop:.3f}")


def plot_class_distribution(df, index_to_label, title="", save_path=None):
    label_counts = df['label'].value_counts().sort_index()
    label_names = [index_to_label[i] for i in label_counts.index]

    # Sort alphabetically
    label_dict = dict(zip(label_names, label_counts))
    sorted_items = sorted(label_dict.items())
    sorted_labels, sorted_counts = zip(*sorted_items)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(sorted_labels), y=list(sorted_counts), palette="Set2")
    plt.title(title, fontsize=22)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(fontsize=20, weight='bold')  # x-axis class labels
    plt.yticks(fontsize=20 )  # y-axis tick labels
    plt.xlabel("Class", fontsize=20, labelpad=10)
    plt.ylabel("Count", fontsize=20, labelpad=5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()