"""
Training & validation loops plus EarlyStopping for CRNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.evals.metrics import ctc_greedy_decode, compute_cer, compute_wer


# ── Helpers ────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best     = float("inf")
        self.counter  = 0

    def step(self, loss: float) -> bool:
        """
        Call after each epoch.

        Returns True when training should stop.
        """
        if loss < self.best:
            self.best    = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler:    torch.cuda.amp.GradScaler,
    idx2char:  dict,
    device:    str  = "cpu",
    use_amp:   bool = False,
    scheduler       = None,
    max_grad_norm: float = 5.0,
) -> tuple[float, float, float]:
    """
    One training epoch with optional AMP and per-batch LR stepping.

    Returns
    -------
    (avg_loss, avg_cer, avg_wer)
    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels, label_lengths, labels_str in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(imgs)                          # (B, T, V)
            T_out   = outputs.size(1)
            input_lengths = torch.full((imgs.size(0),), T_out, dtype=torch.long)
            loss = criterion(
                outputs.permute(1, 0, 2),   # (T, B, V)
                labels, input_lengths, label_lengths,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            all_preds.extend(ctc_greedy_decode(outputs.detach(), idx2char))
        all_targets.extend(labels_str)

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_cer(all_preds, all_targets), compute_wer(all_preds, all_targets)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(
    model:    nn.Module,
    loader,
    criterion: nn.Module,
    idx2char:  dict,
    device:    str  = "cpu",
    use_amp:   bool = False,
) -> tuple[float, float, float]:
    """
    Validation pass.

    Returns
    -------
    (avg_loss, avg_cer, avg_wer)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels, label_lengths, labels_str in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device, enabled=use_amp):
                outputs = model(imgs)
                T_out   = outputs.size(1)
                input_lengths = torch.full((imgs.size(0),), T_out, dtype=torch.long)
                loss = criterion(
                    outputs.permute(1, 0, 2),
                    labels, input_lengths, label_lengths,
                )

            total_loss += loss.item()
            all_preds.extend(ctc_greedy_decode(outputs.float(), idx2char))
            all_targets.extend(labels_str)

    avg_loss = total_loss / len(loader)
    return avg_loss, compute_cer(all_preds, all_targets), compute_wer(all_preds, all_targets)


# ── TrOCR fine-tuning ──────────────────────────────────────────────────────────

def train_trocr(
    dataset_dir:    str,
    save_dir:       str,
    base_model:     str   = "microsoft/trocr-base-printed",
    epochs:         int   = 30,
    batch_size:     int   = 4,
    lr:             float = 5e-5,
    max_target_len: int   = 128,
    es_patience:    int   = 7,
    device:         str   = "cpu",
    use_amp:        bool  = False,
    num_workers:    int   = 2,
    seed:           int   = 42,
    plot_save_path: str | None = None,
) -> dict:
    """
    Fine-tune TrOCR on a line-image dataset.

    Memory management
    -----------------
    - Gradient checkpointing is enabled on GPU to trade compute for VRAM.
    - GPU cache is cleared after every epoch.
    - Python GC is forced after every epoch.
    - Call ``del`` on the returned dict and ``torch.cuda.empty_cache()``
      after you're done if immediately switching to inference.

    Parameters
    ----------
    dataset_dir    : root folder containing book sub-dirs with labels.csv
    save_dir       : directory where the fine-tuned model is saved
    base_model     : HuggingFace model id to fine-tune from
    epochs         : max training epochs
    batch_size     : per-GPU batch size
    lr             : peak learning rate (OneCycleLR)
    max_target_len : max tokenised label length
    es_patience    : early-stopping patience (epochs)
    device         : 'cpu' or 'cuda'
    use_amp        : enable mixed-precision (requires CUDA)
    num_workers    : DataLoader workers
    seed           : random seed for train/val split
    plot_save_path : if provided, save training-curve plot to this path

    Returns
    -------
    dict with keys: train_losses, val_losses, val_cers, val_wers, best_loss
    """
    import gc
    import os
    import torch
    from torch.utils.data import DataLoader, random_split
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from transformers import logging as hf_logging
    from tqdm import tqdm

    from src.modelutils.dataset import TrOCRLineDataset, trocr_collate
    from src.evals.metrics import compute_cer, compute_wer

    os.makedirs(save_dir, exist_ok=True)

    # ── Load processor + model ────────────────────────────────────────────────
    hf_logging.set_verbosity_error()
    print(f"Loading  {base_model} ...")
    processor = TrOCRProcessor.from_pretrained(base_model)
    model     = VisionEncoderDecoderModel.from_pretrained(base_model)
    hf_logging.set_verbosity_warning()

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model = model.to(device)

    if use_amp:
        model.encoder.gradient_checkpointing_enable()

    # ── Dataset split ─────────────────────────────────────────────────────────
    full_ds = TrOCRLineDataset(dataset_dir, processor, max_target_len=max_target_len)
    n_train = int(0.9 * len(full_ds))
    n_val   = len(full_ds) - n_train
    _gen    = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=_gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=trocr_collate, pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=trocr_collate, pin_memory=use_amp,
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler     = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stop = EarlyStopping(es_patience)

    train_losses, val_losses = [], []
    val_cers,     val_wers   = [], []
    best_loss = float("inf")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrOCR params: {n_params:,}  |  device: {device}  |  AMP: {use_amp}")

    bar = tqdm(range(1, epochs + 1), desc="TrOCR Fine-tune", unit="ep", dynamic_ncols=True)
    for ep in bar:

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            lb = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = model(pixel_values=pv, labels=lb).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pv  = batch["pixel_values"].to(device, non_blocking=True)
                lb  = batch["labels"].to(device, non_blocking=True)
                gts = batch["texts"]
                with torch.autocast(device_type=device, enabled=use_amp):
                    v_loss += model(pixel_values=pv, labels=lb).loss.item()
                gen_ids = model.generate(pv, max_new_tokens=max_target_len, num_beams=1)
                all_preds.extend(processor.batch_decode(gen_ids, skip_special_tokens=True))
                all_targets.extend(gts)
        v_loss /= len(val_loader)
        v_cer   = compute_cer(all_preds, all_targets)
        v_wer   = compute_wer(all_preds, all_targets)

        train_losses.append(t_loss); val_losses.append(v_loss)
        val_cers.append(v_cer);      val_wers.append(v_wer)

        saved = ""
        if v_loss < best_loss:
            best_loss = v_loss
            saved = " ✓"
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)

        bar.set_postfix(
            tL=f"{t_loss:.4f}", vL=f"{v_loss:.4f}",
            vCER=f"{v_cer:.3f}", vWER=f"{v_wer:.3f}",
            best=f"{best_loss:.4f}", saved=saved,
        )

        # Memory cleanup per epoch
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if early_stop.step(v_loss):
            bar.write(f"\nEarly stopping at epoch {ep}.")
            break

    bar.close()
    print(f"\nTrOCR fine-tuning complete.  Best val loss: {best_loss:.4f}")
    print(f"Model saved → {save_dir}")

    if plot_save_path:
        from src.plots.training import plot_training_history
        plot_training_history(
            train_losses, val_losses,
            [0.0] * len(val_cers), val_cers,
            [0.0] * len(val_wers), val_wers,
            save_path=plot_save_path,
            title="TrOCR Training History",
        )

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "val_cers":     val_cers,
        "val_wers":     val_wers,
        "best_loss":    best_loss,
    }
