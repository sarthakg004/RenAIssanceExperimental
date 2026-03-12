##########################################################
# Layout detection utils
##########################################################

import cv2
import matplotlib.pyplot as plt
import gc
import paddle

from paddleocr import PPStructureV3


# ---------------------------------------------------
# Resize large image for layout detection
# ---------------------------------------------------
def resize_for_layout(image, max_side=1800):
    h, w = image.shape[:2] 
    max_dim = max(h, w) 
    if max_dim <= max_side: return image, 1.0 
    scale = max_side / max_dim 
    new_w = int(w * scale) 
    new_h = int(h * scale) 
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) 
    return resized, scale


# ---------------------------------------------------
# Box utilities
# ---------------------------------------------------

def box_area(box):
    x1, y1, x2, y2 = box
    return (x2-x1)*(y2-y1)


def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB-xA)
    inter_h = max(0, yB-yA)

    inter = inter_w * inter_h

    if inter == 0:
        return 0

    areaA = box_area(boxA)
    areaB = box_area(boxB)

    return inter / float(areaA + areaB - inter)


# ---------------------------------------------------
# Remove overlapping layouts (keep smaller)
# ---------------------------------------------------

def suppress_overlapping(layout_boxes, iou_thresh=0.3):

    filtered = []

    for box in layout_boxes:

        keep = True

        for kept in filtered:

            if box["label"] != kept["label"]:
                continue

            overlap = iou(box["bbox"], kept["bbox"])

            if overlap > iou_thresh:

                if box_area(box["bbox"]) < box_area(kept["bbox"]):
                    filtered.remove(kept)
                else:
                    keep = False

                break

        if keep:
            filtered.append(box)

    return filtered


# ---------------------------------------------------
# Remove margin text
# ---------------------------------------------------

def remove_margin_boxes(layout_boxes, page_width):

    filtered = []

    for b in layout_boxes:

        x1,y1,x2,y2 = b["bbox"]

        center_x = (x1 + x2) / 2

        if center_x < page_width * 0.12:
            continue

        if center_x > page_width * 0.88:
            continue

        filtered.append(b)

    return filtered


# ---------------------------------------------------
# Merge title fragments
# ---------------------------------------------------

def merge_title_blocks(layout_boxes, vertical_thresh=70):

    merged = []
    
    # process each label separately
    for label in ["doc_title", "text"]:

        boxes = [b for b in layout_boxes if b["label"] == label]
        boxes = sorted(boxes, key=lambda x: x["bbox"][1])

        current = None

        for b in boxes:

            if current is None:
                current = b.copy()
                continue

            gap = b["bbox"][1] - current["bbox"][3]

            if gap < vertical_thresh:

                x1 = min(current["bbox"][0], b["bbox"][0])
                y1 = min(current["bbox"][1], b["bbox"][1])
                x2 = max(current["bbox"][2], b["bbox"][2])
                y2 = max(current["bbox"][3], b["bbox"][3])

                current["bbox"] = [x1, y1, x2, y2]

            else:
                merged.append(current)
                current = b.copy()

        if current:
            merged.append(current)

    # add all other labels unchanged
    for b in layout_boxes:
        if b["label"] not in ["doc_title", "text"]:
            merged.append(b)

    return merged

# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------

def detect_layout(
        image,
        layout_model="PP-DocLayout_plus-L",
        visualize=False,
        max_layout_side=1200,
        device="gpu"
    ):

    resized_img, scale = resize_for_layout(image, max_layout_side)

    pipeline = PPStructureV3(
        layout_detection_model_name=layout_model,
        device=device,

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_region_detection=False,
    )

    results = pipeline.predict(input=resized_img)

    layout_boxes = []

    for res in results:

        boxes = res["layout_det_res"]["boxes"]

        for b in boxes:

            label = b["label"]
            score = float(b["score"])

            x1, y1, x2, y2 = b["coordinate"]

            ox1 = int(x1 / scale)
            oy1 = int(y1 / scale)
            ox2 = int(x2 / scale)
            oy2 = int(y2 / scale)

            layout_boxes.append({
                "label": label,
                "bbox": [ox1, oy1, ox2, oy2],
                "score": score
            })

    # -----------------------------------
    # Post-processing pipeline
    # -----------------------------------

    layout_boxes = suppress_overlapping(layout_boxes)

    layout_boxes = remove_margin_boxes(layout_boxes, image.shape[1])

    layout_boxes = merge_title_blocks(layout_boxes)

    # -----------------------------------
    # Visualization
    # -----------------------------------

    if visualize:

        vis = image.copy()

        color_map = {
            "text": (0,255,0),
            "doc_title": (255,0,0),
            "paragraph_title": (255,255,0),
                "header": (0,255,255),
                "outside_text": (255,0,255)
        }

        for box in layout_boxes:

            if box["label"] not in ["header","paragraph_title","text","doc_title","outside_text"]:
                continue

            x1,y1,x2,y2 = box["bbox"]

            color = color_map[box["label"]]

            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                vis,
                box["label"],
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        plt.figure(figsize=(12,14))
        plt.title("Text + Title Layout")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    # -----------------------------------
    # Memory cleanup
    # -----------------------------------

    del pipeline
    del results
    gc.collect()

    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()

    return layout_boxes

############################################################
# paddle ocr utils
###########################################################
import numpy as np
# ── Config ────────────────────────────────────────────────────────────────
TEXT_LABELS       = {'text', 'paragraph_title', 'doc_title', 'header'}
REGION_PADDING    = 40     # white pixels added around every crop
LAYOUT_EXPAND     = 5     # slight outward expansion of each layout bbox
SCORE_THRESH      = 0.5    # discard raw OCR boxes below this confidence
UPSCALE_MIN_H     = 60     # upscale crop 2× when its height is below this (px)
NMS_IOU_THRESH    = 0.3    # IoU threshold for NMS deduplication
GAP_MULTIPLIER    = 2.0    # horizontal gap limit = GAP_MULTIPLIER × median_char_width

def _poly_bounds(poly):
    """Return (x_min, y_min, x_max, y_max, cx, cy, w, h) for a 4-pt polygon."""
    pts = np.asarray(poly, dtype=np.float32)
    xmn, ymn = pts[:, 0].min(), pts[:, 1].min()
    xmx, ymx = pts[:, 0].max(), pts[:, 1].max()
    return xmn, ymn, xmx, ymx, (xmn+xmx)*0.5, (ymn+ymx)*0.5, xmx-xmn, ymx-ymn


def _iou(a, b):
    """IoU between two (xmn, ymn, xmx, ymx) tuples."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter)


def _nms(boxes, scores, iou_thresh=NMS_IOU_THRESH):
    """
    Non-maximum suppression on axis-aligned boxes.
    Keeps higher-scored box when two boxes overlap > iou_thresh.
    Returns indices of surviving boxes.
    """
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep  = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        bi = (_poly_bounds(boxes[i])[:4])
        for j in order:
            if j in suppressed or j == i:
                continue
            bj = (_poly_bounds(boxes[j])[:4])
            if _iou(bi, bj) > iou_thresh:
                suppressed.add(j)
    return keep


def _resolve_vertical_overlaps(boxes, thresh=0.30):
    """
    When two line boxes overlap vertically by more than `thresh` of
    the smaller box's height, trim both to meet at the midpoint of
    their overlap zone. All lines are preserved.
    """
    if len(boxes) < 2:
        return boxes

    rects = [
        [float(b[:,0].min()), float(b[:,1].min()),
         float(b[:,0].max()), float(b[:,1].max())]
        for b in boxes
    ]
    order = sorted(range(len(rects)), key=lambda i: rects[i][1])
    rects = [rects[i] for i in order]

    for i in range(len(rects)):
        yi0, yi1 = rects[i][1], rects[i][3]
        hi = yi1 - yi0
        if hi <= 0:
            continue
        for j in range(i+1, len(rects)):
            yj0, yj1 = rects[j][1], rects[j][3]
            ov_top = max(yi0, yj0)
            ov_bot = min(yi1, yj1)
            ov     = ov_bot - ov_top
            if ov <= 0:
                break
            hj = yj1 - yj0
            if hj <= 0:
                continue
            if ov / min(hi, hj) > thresh:
                mid        = (ov_top + ov_bot) / 2.0
                rects[i][3] = mid
                rects[j][1] = mid
                yi1         = mid

    result = []
    for r in rects:
        xmn, ymn, xmx, ymx = r
        if xmx > xmn and ymx > ymn:
            result.append(np.array(
                [[xmn,ymn],[xmx,ymn],[xmx,ymx],[xmn,ymx]],
                dtype=np.float32
            ))
    return result


def _merge_into_lines(raw_boxes, img_w, img_h):
    """
    Group word polygons into line-level axis-aligned boxes.

    Steps
    -----
    1. Compute statistics → height_thresh, gap_limit.
    2. Filter tiny / giant boxes.
    3. Sort by center_y; greedily assign to nearest open line.
    4. Within each line sort left→right; merge with gap < GAP_MULTIPLIER × median_w.
    5. Resolve vertical overlaps by trimming to midpoint.
    6. Sort final boxes into reading order (top→bottom, left→right).
    """
    if not raw_boxes:
        return []

    bounds  = [_poly_bounds(b) for b in raw_boxes]
    heights = np.array([b[7] for b in bounds], dtype=np.float32)
    widths  = np.array([b[6] for b in bounds], dtype=np.float32)

    med_h     = float(np.median(heights))
    med_w     = float(np.median(widths))
    h_thresh  = 0.5  * med_h
    gap_limit = GAP_MULTIPLIER * med_w
    MIN_W, MIN_H, MAX_H = 10.0, 10.0, 3.0 * med_h

    filtered = [
        (b, bnd) for b, bnd in zip(raw_boxes, bounds)
        if bnd[6] >= MIN_W and MIN_H <= bnd[7] <= MAX_H
    ]
    if not filtered:
        return []

    filtered.sort(key=lambda x: x[1][5])   # sort by center_y

    lines = []
    for box, bnd in filtered:
        cy       = bnd[5]
        assigned = False
        for line in lines:
            line_cy = sum(b[1][5] for b in line) / len(line)
            if abs(cy - line_cy) < h_thresh:
                line.append((box, bnd))
                assigned = True
                break
        if not assigned:
            lines.append([(box, bnd)])

    merged = []
    for line in lines:
        line.sort(key=lambda x: x[1][0])   # left → right
        groups = [[line[0]]]
        for item in line[1:]:
            if item[1][0] - groups[-1][-1][1][2] <= gap_limit:
                groups[-1].append(item)
            else:
                groups.append([item])
        for grp in groups:
            xmn = max(0,     min(b[1][0] for b in grp))
            ymn = max(0,     min(b[1][1] for b in grp))
            xmx = min(img_w, max(b[1][2] for b in grp))
            ymx = min(img_h, max(b[1][3] for b in grp))
            if xmx > xmn and ymx > ymn:
                merged.append(np.array(
                    [[xmn,ymn],[xmx,ymn],[xmx,ymx],[xmn,ymx]],
                    dtype=np.float32
                ))

    merged = _resolve_vertical_overlaps(merged)

    # Reading order: top → bottom, then left → right within same row
    merged.sort(key=lambda b: (float(b[:,1].min()), float(b[:,0].min())))
    return merged


# ── Page-relative noise filter ────────────────────────────────────────────

def filter_boxes_by_page_size(
    boxes: list,
    img_h: int,
    img_w: int,
    min_area_fraction: float = 0.0001,
) -> list:
    """
    Remove line boxes whose axis-aligned area is smaller than
    ``min_area_fraction`` × (page height × page width).

    This avoids hard-coded pixel thresholds: the same fraction works
    correctly for both high-resolution scans and smaller images.

    Parameters
    ----------
    boxes             : list of (4, 2) float32 arrays in page coordinates
    img_h, img_w      : page dimensions in pixels
    min_area_fraction : fraction of page area used as the lower bound
                        (default 0.0001 = 0.01 % of the page)

    Returns
    -------
    Filtered list of boxes.
    """
    page_area = img_h * img_w
    min_area  = min_area_fraction * page_area
    kept = []
    for box in boxes:
        pts = np.asarray(box, dtype=np.float32)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        if w * h >= min_area:
            kept.append(box)
    return kept


# ── Main function ─────────────────────────────────────────────────────────

def detect_text_lines(image, layout, ocr,
                      text_labels=None,
                      region_padding=REGION_PADDING,
                      layout_expand=LAYOUT_EXPAND,
                      score_thresh=SCORE_THRESH,
                      upscale_min_h=UPSCALE_MIN_H,
                      min_area_fraction: float = 0.0001):
    """
    Detect line-level bounding boxes for all text regions on a page.

    Improvements applied
    --------------------
    - Score threshold filtering   : discard raw boxes with score < score_thresh
    - NMS deduplication           : remove duplicate boxes at region boundaries
    - Small-region upscaling      : 2× upscale crops shorter than upscale_min_h px
    - Tighter horizontal merging  : gap_limit = GAP_MULTIPLIER × median_char_width
    - Page-relative noise filter  : drop merged boxes smaller than
                                    min_area_fraction × page area (adaptive,
                                    no hard-coded pixel threshold)
    - Reading-order sort          : output sorted top→bottom, left→right

    Parameters
    ----------
    min_area_fraction : float
        Minimum merged-box area as a fraction of the full page area.
        Defaults to 0.0001 (0.01 %).  Increase to be more aggressive.

    Returns
    -------
    merged_boxes : list[np.ndarray]  each shape (4, 2) float32, page coords
    """
    if text_labels is None:
        text_labels = TEXT_LABELS

    img_h, img_w = image.shape[:2]
    all_raw    = []
    all_scores = []

    for region in layout:
        if region['label'] not in text_labels:
            continue

        x1, y1, x2, y2 = [int(c) for c in region['bbox']]
        x1e = max(0,     x1 - layout_expand)
        y1e = max(0,     y1 - layout_expand)
        x2e = min(img_w, x2 + layout_expand)
        y2e = min(img_h, y2 + layout_expand)

        if x2e <= x1e or y2e <= y1e:
            continue

        crop = image[y1e:y2e, x1e:x2e]
        if crop.size == 0:
            continue

        # ── Upscale small crops ───────────────────────────────────────────
        scale  = 1.0
        crop_h = crop.shape[0]
        if crop_h < upscale_min_h:
            scale = 2.0
            crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # ── White border ──────────────────────────────────────────────────
        pad_scaled = int(region_padding * scale)
        padded = cv2.copyMakeBorder(
            crop,
            top=pad_scaled, bottom=pad_scaled,
            left=pad_scaled, right=pad_scaled,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        del crop

        # Padded-crop origin in page space
        ox = x1e - region_padding
        oy = y1e - region_padding

        crop_results = ocr.predict(padded)
        del padded

        n = 0
        for res in crop_results:
            polys  = res.get('dt_polys',  [])
            scores = res.get('dt_scores', [1.0] * len(polys))
            for poly, score in zip(polys, scores):
                # ── Score filter ──────────────────────────────────────────
                if score < score_thresh:
                    continue
                arr = np.array(poly, dtype=np.float32)
                # Undo upscaling, then shift to page coordinates
                arr /= scale
                arr[:, 0] += ox
                arr[:, 1] += oy
                # Discard detections wholly inside the white border
                if arr[:, 0].max() < x1e or arr[:, 0].min() > x2e:
                    continue
                if arr[:, 1].max() < y1e or arr[:, 1].min() > y2e:
                    continue
                all_raw.append(arr)
                all_scores.append(float(score))
                n += 1
        del crop_results

        print(f"  [{region['label']}]  bbox={region['bbox']}  "
              f"scale={scale:.1f}x  →  {n} raw boxes (after score filter)")

    print(f"\nRaw boxes before NMS : {len(all_raw)}")

    # ── NMS deduplication ─────────────────────────────────────────────────
    keep_idx = _nms(all_raw, all_scores)
    all_raw  = [all_raw[i]    for i in keep_idx]
    print(f"Raw boxes after NMS  : {len(all_raw)}")

    merged = _merge_into_lines(all_raw, img_w, img_h)
    del all_raw

    # ── Page-relative noise removal ───────────────────────────────────────
    before  = len(merged)
    merged  = filter_boxes_by_page_size(merged, img_h, img_w, min_area_fraction)
    removed = before - len(merged)
    print(f"Boxes after page-relative filter : {len(merged)}  "
          f"(removed {removed} ← < {min_area_fraction*100:.3f}% of page area)")

    print(f"Line-level boxes     : {len(merged)}")
    return merged

############################################################################
#  DataSet utils
############################################################################
import cv2
import re
import os
import sys
import numpy as np
import warnings
import contextlib
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper: match transcript folder for a given image book folder
# ---------------------------------------------------------------------------
def match_transcript_folder(book_folder_name: str, transcript_root_dir: str | Path) -> Path | None:
    """
    Given a book folder name (e.g. 'PORCONES.23.5 - 1628'), return the path
    to the matching transcript folder inside transcript_root_dir.
    Convention: transcript folder = book_folder_name + ' transcription'
    Returns None if no match is found.
    """
    transcript_root = Path(transcript_root_dir)
    candidate = transcript_root / (book_folder_name + " transcription")
    if candidate.is_dir():
        return candidate
    book_lower = (book_folder_name + " transcription").lower()
    for entry in transcript_root.iterdir():
        if entry.is_dir() and entry.name.lower() == book_lower:
            return entry
    return None


# ---------------------------------------------------------------------------
# Helper: sort line boxes in natural reading order (top→bottom, left→right)
# ---------------------------------------------------------------------------
def sort_line_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
    """Sort quadrilateral polygon boxes by centroid: y first, then x."""
    def centroid(box):
        pts = np.array(box, dtype=np.float32)
        return pts[:, 1].mean(), pts[:, 0].mean()  # (cy, cx)
    return sorted(boxes, key=centroid)


# ---------------------------------------------------------------------------
# Helper: convert transcript text to a safe filename stem
# ---------------------------------------------------------------------------
def sanitize_label(text: str, max_len: int = 80) -> str:
    """
    Turn a transcript line into a filesystem-safe filename stem.
    Spaces become underscores; forbidden characters are stripped.
    """
    safe = re.sub(r'[\\/:*?"<>|\n\r\t]', '', text)
    safe = safe.strip().replace(' ', '_')
    safe = re.sub(r'_+', '_', safe)          # collapse multiple underscores
    return safe[:max_len] if safe else "unknown"


# ---------------------------------------------------------------------------
# Helper: crop a line image using a polygon's axis-aligned bounding box
# ---------------------------------------------------------------------------
def crop_line_from_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    padding: int = 5,
) -> np.ndarray | None:
    """
    Crop the axis-aligned bounding box of a quadrilateral from the page image.
    Returns None if the crop area is degenerate.
    """
    pts = np.array(polygon, dtype=np.int32)
    h, w = image.shape[:2]
    x_min = max(0, pts[:, 0].min() - padding)
    y_min = max(0, pts[:, 1].min() - padding)
    x_max = min(w, pts[:, 0].max() + padding)
    y_max = min(h, pts[:, 1].max() + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]



def build_ocr_dataset(
    image_dir: str | Path,
    transcript_dir: str | Path,
    output_dir: str | Path,
    ocr,
    layout_model: str = "PP-DocLayout_plus-L",
    max_layout_side: int = 1500,
    padding: int = 5
) -> None:
    """
    Build a line-level OCR training dataset for a single book.

    Output structure:
        output_dir/
            page1/
                0001_AL_CORREGIDOR.png
                0002_DE_LOS_INDIOS.png
            page2/
                0001_POR.png
                ...

    Filename format:  {line_no:04d}_{sanitized_label}.png
    Duplicate labels within the same page get a --n suffix.

    Parameters
    ----------
    image_dir       : folder with page images for one book
    transcript_dir  : folder with matching per-page .txt transcript files
    output_dir      : root destination; one sub-folder is created per page
    ocr             : initialised PaddleOCR instance
    layout_model    : layout detection model name
    max_layout_side : resize limit for layout detection
    padding         : pixel padding around each cropped line
    min_box_area    : drop detected boxes smaller than this (px²) — removes noise
    """
    image_dir = Path(image_dir)
    transcript_dir = Path(transcript_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    book_name = image_dir.name

    # Collect page pairs
    page_pairs: list[tuple[Path, Path]] = []
    for img_file in sorted(image_dir.iterdir()):
        if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        transcript_file = transcript_dir / (img_file.stem + ".txt")
        if not transcript_file.exists():
            warnings.warn(f"[SKIP] Transcript missing: {transcript_file}")
            continue
        page_pairs.append((img_file, transcript_file))

    if not page_pairs:
        print(f"No page pairs found for '{book_name}' — check image_dir and transcript_dir.")
        return

    total_saved = 0
    total_skipped_noise = 0

    bar = tqdm(page_pairs, desc=book_name, unit="page", dynamic_ncols=True)
    for img_path, transcript_path in bar:

        # ── 1. Load image ──────────────────────────────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            warnings.warn(f"[SKIP] Cannot read image: {img_path}")
            continue

        # ── 2. Load transcript ─────────────────────────────────────────────
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_lines = [l.strip() for l in f if l.strip()]
        except Exception as exc:
            warnings.warn(f"[SKIP] Cannot read transcript {transcript_path}: {exc}")
            del image; continue

        if not transcript_lines:
            warnings.warn(f"[SKIP] Empty transcript: {transcript_path}")
            del image; continue

        # ── 3. Layout detection (output suppressed) ────────────────────────
        try:
            layout = detect_layout(image, layout_model, visualize=False)
        except Exception as exc:
            warnings.warn(f"[SKIP] Layout detection failed for {img_path.name}: {exc}")
            del image; continue

        if not layout:
            warnings.warn(f"[SKIP] No layout regions found: {img_path.name}")
            del image; continue

        # ── 4. Text line detection (output suppressed) ─────────────────────
        try:
            merged_boxes = detect_text_lines(image, layout, ocr)
        except Exception as exc:
            warnings.warn(f"[SKIP] Text line detection failed for {img_path.name}: {exc}")
            del image; del layout; continue

        if not merged_boxes:
            warnings.warn(f"[SKIP] No text lines detected: {img_path.name}")
            del image; del layout; continue

        # ── 5. Filter small / noise boxes ─────────────────────────────────
        before = len(merged_boxes)
        noise_removed = before - len(merged_boxes)
        total_skipped_noise += noise_removed

        if not merged_boxes:
            warnings.warn(f"[SKIP] All boxes filtered as noise: {img_path.name}")
            del image; del layout; continue

        # ── 6. Sort & align ────────────────────────────────────────────────
        sorted_boxes = sort_line_boxes(merged_boxes)
        num_pairs = min(len(sorted_boxes), len(transcript_lines))
        if len(sorted_boxes) != len(transcript_lines):
            warnings.warn(
                f"[MISMATCH] {img_path.name}: "
                f"{len(sorted_boxes)} boxes vs {len(transcript_lines)} lines "
                f"— using first {num_pairs}."
            )

        # ── 7. Create per-page output directory ────────────────────────────
        page_out_dir = out_root / img_path.stem   # e.g.  output_dir/page1/
        page_out_dir.mkdir(parents=True, exist_ok=True)

        # Track duplicate label stems within this page
        seen_labels: dict[str, int] = {}
        lines_saved = 0

        # ── 8. Crop & save ─────────────────────────────────────────────────
        for i in range(num_pairs):
            line_no = i + 1                        # 1-based line number
            text = transcript_lines[i]
            line_img = crop_line_from_polygon(image, sorted_boxes[i], padding=padding)
            if line_img is None or line_img.size == 0:
                continue

            # Build filename:  {line_no:04d}_{sanitized_label}[--n].png
            stem = sanitize_label(text)
            count = seen_labels.get(stem, 0) + 1
            seen_labels[stem] = count
            label_part = stem if count == 1 else f"{stem}--{count}"
            filename = f"{line_no:04d}_{label_part}.png"

            cv2.imwrite(str(page_out_dir / filename), line_img)
            lines_saved += 1
            del line_img

        total_saved += lines_saved

        bar.set_postfix(
            page=img_path.name,
            saved=lines_saved,
            noise_removed=noise_removed,
            refresh=True,
        )

        # ── 9. Memory cleanup ───────────────────────────────────────────────
        del image, layout, merged_boxes, sorted_boxes

    bar.close()
    print(f"\nDone.")
    print(f"  Total line images saved : {total_saved}")
    print(f"  Noise boxes removed     : {total_skipped_noise}")
    print(f"  Output root             : {out_root}")
