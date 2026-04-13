import json
import torch
from sklearn.metrics import classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_grids(history_name):
    file_path = Path(__file__).parent.parent.parent / "model" / history_name
    with open(file_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    train_acc, train_loss = history["train_acc"], history["train_loss"]
    test_acc, test_loss = history["test_acc"], history["test_loss"]

    epochs = range(len(train_acc))
    data = pd.DataFrame({
        'epoch': list(epochs) * 2,
        'Accuracy': train_acc + test_acc,
        'Loss': train_loss + test_loss,
        'Dataset': ['Train'] * len(train_acc) + ['Test'] * len(test_acc)
    })

    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=data, x='epoch', y='Accuracy', hue='Dataset', ax=ax1, marker='o')
    ax1.set_title('Model Accuracy')

    sns.lineplot(data=data, x='epoch', y='Loss', hue='Dataset', ax=ax2, marker='o')
    ax2.set_title('Model Loss')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_grids("history_model2.json")
