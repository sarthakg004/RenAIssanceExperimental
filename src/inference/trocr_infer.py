"""
Full-page OCR inference using a fine-tuned TrOCR model.

Usage
-----
    from src.inference.trocr_infer import trocr_infer_page

    lines = trocr_infer_page("data/3.processed/page1.png",
                              model_dir="checkpoints/trocr")
    for i, line in enumerate(lines, 1):
        print(f"[{i:3d}] {line}")
"""

from __future__ import annotations

import gc
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.inference.utils import crop_polygon_rgb


def trocr_infer_page(
    image_path:        str,
    model_dir:         str,
    device:            str   = "cpu",
    layout_model:      str   = "PP-DocLayout_plus-L",
    det_model:         str   = "PP-OCRv5_server_det",
    rec_model:         str   = "PP-OCRv5_server_rec",
    max_layout_side:   int   = 1500,
    min_area_fraction: float = 0.0001,
    num_beams:         int   = 4,
    max_new_tokens:    int   = 128,
    visualize:         bool  = True,
) -> list[str]:
    """
    Full-page OCR: layout detection → layout-aware line detection → TrOCR recognition.

    Memory management (4 staged)
    ----------------------------
    Stage 1  PPStructureV3 layout model detects page regions, freed internally
             by detect_layout() before returning.
    Stage 2  PaddleOCR line-detection model is loaded, run per layout region,
             then explicitly deleted + GPU memory freed before TrOCR loads.
    Stage 3  TrOCR is loaded, all lines are recognised, then freed.
    Stage 4  Visualisation (no model in memory).

    Parameters
    ----------
    image_path         : path to the full-page image.
    model_dir          : directory written by ``save_pretrained``.
    device             : 'cpu' or 'cuda'.
    layout_model       : PaddlePaddle layout-detection model name.
    det_model          : PaddleOCR text-line detection model name.
    rec_model          : PaddleOCR model name for pipeline init (detection only).
    max_layout_side    : max pixel dimension when resizing for layout detection.
    min_area_fraction  : drop merged line boxes smaller than this fraction of page.
    num_beams          : beam-search width for TrOCR generation.
    max_new_tokens     : max output tokens per line.
    visualize          : show an annotated figure after inference.

    Returns
    -------
    list[str]  — one string per detected text line, in reading order.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    paddle_device = "gpu" if device == "cuda" else "cpu"

    # ── Stage 1: Layout detection ─────────────────────────────────────────────
    # detect_layout() frees PPStructureV3 internally before returning.
    print("[1/4] Running layout detection ...")
    from src.textDetection import detect_layout
    layout = detect_layout(
        image_bgr,
        layout_model     = layout_model,
        visualize        = False,
        max_layout_side  = max_layout_side,
        device           = paddle_device,
    )
    text_region_count = sum(
        1 for r in layout
        if r["label"] in {"text", "paragraph_title", "doc_title", "header"}
    )
    print(f"    └─ Found {text_region_count} text regions.")

    # ── Stage 2: Layout-aware text-line detection ─────────────────────────────
    print("[2/4] Running layout-aware line detection ...")
    from paddleocr import PaddleOCR
    from src.textDetection import detect_text_lines
    _ocr = PaddleOCR(
        text_detection_model_name    = det_model,
        text_recognition_model_name  = rec_model,
        use_doc_orientation_classify = False,
        use_doc_unwarping            = False,
        use_textline_orientation     = False,
        lang                         = "sp",
    )
    all_boxes = detect_text_lines(
        image_bgr, layout, _ocr,
        min_area_fraction = min_area_fraction,
    )
    print(f"    └─ Detected {len(all_boxes)} text lines.")

    # Free the line detector before loading TrOCR
    del _ocr
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Stage 3: TrOCR recognition ────────────────────────────────────────────
    print(f"[3/4] Loading TrOCR from {model_dir} ...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc  = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    transcripts = []
    for box in all_boxes:
        pil_crop     = crop_polygon_rgb(image_bgr, box)
        pixel_values = proc(pil_crop, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values,
                num_beams      = num_beams,
                max_new_tokens = max_new_tokens,
                early_stopping = True,
            )
        transcripts.append(proc.batch_decode(gen_ids, skip_special_tokens=True)[0])

    # Free TrOCR
    del model, proc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"    └─ Recognised {len(transcripts)} lines.")

    # ── Stage 4: Visualise ────────────────────────────────────────────────────
    if visualize:
        print("[4/4] Visualising ...")
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
        # Layout regions — red
        for region in layout:
            if region["label"] not in {"text", "paragraph_title", "doc_title", "header"}:
                continue
            rx1, ry1, rx2, ry2 = [int(c) for c in region["bbox"]]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        # Line boxes — green with transcript overlay
        for box, text in zip(all_boxes, transcripts):
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 200, 80), 2)
            x, y = np.array(box).min(axis=0).astype(int)
            cv2.putText(vis, text[:50], (x, max(y - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 50, 50), 1)
        plt.figure(figsize=(12, 16))
        plt.imshow(vis)
        plt.axis("off")
        plt.title(
            f"TrOCR Inference  |  {text_region_count} layout regions  |  {len(all_boxes)} lines",
            fontsize=13,
        )
        plt.tight_layout()
        plt.show()

    return transcripts
