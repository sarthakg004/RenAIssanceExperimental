"""
Training history plots for OCR models (loss, CER, WER curves).
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt


def plot_training_history(
    train_losses: list[float],
    val_losses:   list[float],
    train_cers:   list[float],
    val_cers:     list[float],
    train_wers:   list[float],
    val_wers:     list[float],
    save_path:  str | None = None,
    title:      str        = "Training History",
) -> None:
    """
    Plot loss, CER and WER curves for train and validation splits.

    Parameters
    ----------
    *_losses / *_cers / *_wers : per-epoch metric lists
    save_path                  : if provided, save figure to this path
    title                      : figure suptitle
    """
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)

    for ax, (tr, va, ylabel, t) in zip(
        axes,
        [
            (train_losses, val_losses, "Loss",  "CTC Loss"),
            (train_cers,   val_cers,   "CER",   "Character Error Rate"),
            (train_wers,   val_wers,   "WER",   "Word Error Rate"),
        ],
    ):
        ax.plot(epochs, tr, label="Train")
        ax.plot(epochs, va, label="Validation")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")

    plt.show()
