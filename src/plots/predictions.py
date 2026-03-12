"""
Sample-prediction grid plot for CRNN OCR models.
"""

from __future__ import annotations
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torchvision.transforms as T

from src.evals.metrics import ctc_greedy_decode
from src.modelutils.dataset import resize_keep_ratio

# ── Colour palette ─────────────────────────────────────────────────────────────
_HIT   = "#27ae60"
_MISS  = "#e74c3c"
_LABEL = "#888888"
_BG    = "#f9f9f9"


# ── Internal helper ────────────────────────────────────────────────────────────

def _draw_word_labels(ax, pred: str, gt: str, fontsize: float = 8.5) -> None:
    """Write colour-coded word-comparison text on *ax*."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    probe = ax.text(0, 0, "M", transform=ax.transAxes,
                    fontsize=fontsize, family="monospace", alpha=0)
    fig.canvas.draw()
    cw = probe.get_window_extent(renderer).width / ax.get_window_extent(renderer).width
    probe.remove()

    def _write_line(words, ref_words, y, prefix):
        x = 0.01
        ax.text(x, y, prefix, transform=ax.transAxes, fontsize=fontsize,
                family="monospace", color=_LABEL, va="center", clip_on=False)
        x += len(prefix) * cw
        for i, word in enumerate(words):
            hit = i < len(ref_words) and word == ref_words[i]
            ax.text(x, y, word, transform=ax.transAxes, fontsize=fontsize,
                    family="monospace", color=_HIT if hit else _MISS,
                    va="center", clip_on=False)
            x += (len(word) + 1) * cw

    _write_line(pred.strip().split(), gt.strip().split(), y=0.72, prefix="Pred: ")
    _write_line(gt.strip().split(),   pred.strip().split(), y=0.28, prefix="GT:   ")


# ── Public function ────────────────────────────────────────────────────────────

def plot_sample_predictions(
    model:        torch.nn.Module,
    val_dataset,
    idx2char:     dict,
    checkpoint:   dict | None  = None,
    img_height:   int          = 64,
    device:       str          = "cpu",
    num_samples:  int          = 6,
    save_path:  str | None     = None,
) -> None:
    """
    Draw a grid of *num_samples* validation line images with
    predicted / ground-truth text shown below each image.

    Parameters
    ----------
    model       : trained CRNN (already .eval() if desired)
    val_dataset : a torch Subset wrapping a LineDataset
    idx2char    : index → character mapping
    checkpoint  : optional dict returned by torch.load() — used for title info
    img_height  : height used during preprocessing
    device      : 'cpu' or 'cuda'
    num_samples : number of samples to display (default 6, arranged 3×2)
    save_path   : if provided, save figure to this path
    """
    model.eval()
    to_tensor = T.ToTensor()

    # Sample from the underlying dataset via the Subset's index map
    base_ds      = val_dataset.dataset
    sample_idxs  = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    sample_data  = [base_ds.samples[val_dataset.indices[i]] for i in sample_idxs]

    # Predict
    results = []
    for img_path, gt_label in sample_data:
        from PIL import Image
        pil       = Image.open(img_path).convert("L")
        pil       = resize_keep_ratio(pil, img_height)
        inp       = to_tensor(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(inp)
        pred = ctc_greedy_decode(log_probs, idx2char)[0]
        results.append((np.array(pil), pred, gt_label, pred.strip() == gt_label.strip()))

    # Build title
    if checkpoint is not None:
        title = (
            f"Sample Predictions  ·  Epoch {checkpoint.get('epoch', '?')}  |  "
            f"val_loss: {checkpoint.get('val_loss', 0):.4f}  "
            f"val_CER: {checkpoint.get('val_cer', 0):.4f}  "
            f"val_WER: {checkpoint.get('val_wer', 0):.4f}"
        )
    else:
        title = "Sample Predictions"

    sns.set_theme(style="white")
    ROWS, COLS = (num_samples + 1) // 2, 2
    fig = plt.figure(figsize=(17, 4 * ROWS), facecolor=_BG)
    fig.suptitle(title, fontsize=12, fontweight="bold", color="#222222", y=1.01)

    gs = gridspec.GridSpec(
        ROWS * 2, COLS,
        height_ratios=[4, 1] * ROWS,
        hspace=0.08, wspace=0.12,
        left=0.03, right=0.97, top=0.93, bottom=0.03,
    )

    for idx, (img_arr, pred, gt, correct) in enumerate(results):
        row, col = divmod(idx, COLS)
        img_row  = row * 2

        ax_img = fig.add_subplot(gs[img_row, col])
        ax_img.imshow(img_arr, cmap="gray")
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_facecolor(_BG)
        border_col = _HIT if correct else _MISS
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.2)
        ax_img.text(0.01, 0.97, f"#{idx+1}", transform=ax_img.transAxes,
                    fontsize=7.5, color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#333333", ec="none"))

        ax_lbl = fig.add_subplot(gs[img_row + 1, col])
        ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
        ax_lbl.axis("off"); ax_lbl.set_facecolor(_BG)
        _draw_word_labels(ax_lbl, pred, gt)
        ax_lbl.axhline(0.97, color="#dddddd", lw=0.8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"Plot saved → {save_path}")

    plt.show()
