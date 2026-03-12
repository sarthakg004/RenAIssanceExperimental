################################
# PDF to Images Conversion
################################
import os
import fitz
from PIL import Image
import re
from pathlib import Path
from tqdm import tqdm
import fitz
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def pdf_to_images(
    pdf_path,
    output_dir,
    book_name=None,
    page_range=None,
    page_list=None,
):
    """
    Convert selected PDF pages to images.

    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Root output directory
        book_name (str): Folder name for this book
        page_range (tuple): (start, end) inclusive page range (1-indexed)
        page_list (list): explicit page numbers (1-indexed)
    """

    # -------------------------
    # Determine book name
    # -------------------------

    if book_name is None:
        book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    book_folder = os.path.join(output_dir, book_name)
    os.makedirs(book_folder, exist_ok=True)

    pdf_doc = fitz.open(pdf_path)
    total_pages = len(pdf_doc)

    # -------------------------
    # Determine pages
    # -------------------------

    if page_list is not None:

        pages_to_process = [
            p - 1 for p in page_list
            if 1 <= p <= total_pages
        ]

    elif page_range is not None:

        start, end = page_range

        if start > end:
            start, end = end, start

        start = max(1, start)
        end = min(total_pages, end)

        # inclusive range
        pages_to_process = list(range(start - 1, end))

    else:
        pages_to_process = list(range(total_pages))

    # Remove duplicates + sort
    pages_to_process = sorted(set(pages_to_process))

    # -------------------------
    # Progress bar
    # -------------------------

    pbar = tqdm(
        pages_to_process,
        desc=f"Converting {book_name}",
        total=len(pages_to_process)
    )

    images_written = 0

    # -------------------------
    # Process pages
    # -------------------------

    for page_num in pbar:

        page = pdf_doc.load_page(page_num)

        # ---- OCR-quality rendering (300 DPI) ----
        pix = page.get_pixmap(
            # dpi=300
            )

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        page_id = page_num + 1

        # ---- Keep cropping logic EXACT ----
        if img.width / img.height > 1.19:

            left = img.crop((0, 0, img.width // 2, img.height))
            right = img.crop((img.width // 2, 0, img.width, img.height))

            left.save(
                os.path.join(book_folder, f"page{page_id}_left.png"),
                format="PNG"
            )

            right.save(
                os.path.join(book_folder, f"page{page_id}_right.png"),
                format="PNG"
            )

            images_written += 2

        else:

            img.save(
                os.path.join(book_folder, f"page{page_id}.png"),
                format="PNG"
            )

            images_written += 1

    pdf_doc.close()

    pbar.close()

    print(f"{images_written} images written → {book_folder}")


import re
from pathlib import Path
from tqdm import tqdm


def transcript_to_page_txt(input_path, output_dir, book_name=None):
    """
    Convert transcript files (.txt or .docx) into page-wise txt files.

    Features:
    - Ignores text before first page marker
    - Stops at 'END OF EXTRACT'
    - Handles markers like:
        PDF p1
        PDF p2 - left
        PDF p3 – right
    - Writes one txt per page
    - Updates existing directories if present
    """

    input_path = Path(input_path)

    # -------------------------
    # Determine book name
    # -------------------------

    if book_name is None:
        book_name = input_path.stem

    book_folder = Path(output_dir) / book_name

    if book_folder.exists():
        print(f"Updating existing directory → {book_folder}")
    else:
        print(f"Creating directory → {book_folder}")

    book_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Read transcript
    # -------------------------

    if input_path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(input_path)
        lines = [p.text for p in doc.paragraphs]
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    # -------------------------
    # Page marker regex
    # -------------------------

    page_pattern = re.compile(
        r'^PDF\s*p\s*(\d+)(?:\s*[-–]\s*(left|right))?',
        re.IGNORECASE
    )

    pages = {}
    current_page = None
    started = False

    # -------------------------
    # Parse transcript
    # -------------------------

    for raw_line in tqdm(lines, desc=f"Parsing {book_name}"):

        line = raw_line.rstrip()

        if "END OF EXTRACT" in line.upper():
            break

        match = page_pattern.match(line.strip())

        if match:
            started = True

            page_num = int(match.group(1))
            side = match.group(2)

            if side:
                current_page = f"page{page_num}_{side.lower()}"
            else:
                current_page = f"page{page_num}"

            pages[current_page] = []
            continue

        if not started:
            continue

        if current_page is not None:
            pages[current_page].append(line)

    # -------------------------
    # Write txt files
    # -------------------------

    for page_name, content in tqdm(
        pages.items(),
        desc=f"Writing {book_name}",
        total=len(pages)
    ):

        txt_path = book_folder / f"{page_name}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content).strip() + "\n")

    print(f"{len(pages)} pages written → {book_folder}")


def build_ocr_dataset(
    pdf_path,
    transcript_path,
    images_output_dir,
    transcripts_output_dir,
    book_name=None,
    page_range=None,
    page_list=None,
):
    """
    Master pipeline for building OCR datasets.

    Steps:
    1. Convert PDF → page images
    2. Convert transcript → page txt files
    3. Validate alignment

    Images and transcripts can be stored in different directories.
    """

    pdf_path = Path(pdf_path)
    transcript_path = Path(transcript_path)

    if book_name is None:
        book_name = pdf_path.stem

    print(f"\nBuilding OCR dataset for: {book_name}")

    # ------------------------
    # Step 1: Generate images
    # ------------------------

    print("\nStep 1: Converting PDF → images")

    pdf_to_images(
        pdf_path=str(pdf_path),
        output_dir=str(images_output_dir),
        book_name=book_name,
        page_range=page_range,
        page_list=page_list,
    )

    # ------------------------
    # Step 2: Generate transcripts
    # ------------------------

    print("\nStep 2: Converting transcript → page txt")

    transcript_to_page_txt(
        input_path=str(transcript_path),
        output_dir=str(transcripts_output_dir),
        book_name=book_name,
    )

    # ------------------------
    # Step 3: Validation
    # ------------------------

    image_dir = Path(images_output_dir) / book_name
    transcript_dir = Path(transcripts_output_dir) / book_name

    image_files = {p.stem for p in image_dir.glob("*.png")}
    transcript_files = {p.stem for p in transcript_dir.glob("*.txt")}

    matched = image_files & transcript_files
    missing_transcripts = image_files - transcript_files
    extra_transcripts = transcript_files - image_files

    print("\nDataset summary")
    print("------------------------")
    print(f"Images: {len(image_files)}")
    print(f"Transcripts: {len(transcript_files)}")
    print(f"Aligned pairs: {len(matched)}")

    if missing_transcripts:
        print(f"\nImages without transcripts: {len(missing_transcripts)}")

    if extra_transcripts:
        print(f"\nTranscripts without images: {len(extra_transcripts)}")

    print("\nPipeline finished.")
    
#############################
# Sample Image Display
#############################
import matplotlib.pyplot as plt

def display_sample(image_path):

    if os.path.exists(image_path):
        img = Image.open(image_path)
        plt.figure(figsize=(4, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Sample Image: {os.path.basename(image_path)}")
        plt.show()
    else:
        print(f"Image {os.path.basename(image_path)} not found.\n")
    
##############################
# Preprocessing for OCR
##############################
import os
import cv2
import numpy as np
from PIL import Image
# Optional — only needed for some methods
from skimage import restoration, exposure


def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def correct_skew(image):
    is_color = len(image.shape) == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    angle = cv2.minAreaRect(largest)[-1]

    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def denoise_image(image, method="nlm"):
    image = image.astype(np.uint8)

    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)

    if method == "nlm":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    return image


def denoise_image(image, method="nlm"):
    image = image.astype(np.uint8)

    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)

    if method == "nlm":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    return image


def enhance_contrast(image, method="clahe"):
    if method != "clahe":
        return image

    clahe = cv2.createCLAHE(2.0, (8,8))

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    return clahe.apply(image)


def binarize_image(image, method="otsu"):
    gray = convert_to_grayscale(image)

    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,15,8
        )

    _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary


def morphological_operations(image, operation="open", k=(2,2), iterations=1):
    kernel = np.ones(k, np.uint8)

    ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE
    }

    if operation in ops:
        return cv2.morphologyEx(image, ops[operation], kernel, iterations)

    if operation == "dilate":
        return cv2.dilate(image, kernel, iterations)

    if operation == "erode":
        return cv2.erode(image, kernel, iterations)

    return image


def remove_large_blobs(image, min_area=3000, min_solidity=0.55,
                       max_aspect_ratio=4.0, erosion_ratio=0.35):
    """
    Neutralise large ink blobs by filling only their inner core with white.

    Instead of erasing the entire connected component (which risks removing
    adjacent letters), we:

      1. Classify each large component as a blob using the same three-gate
         test (area + solidity + aspect ratio).
      2. For every confirmed blob, erode its mask by a fraction of its own
         size to obtain only the safe interior core.
      3. Paint that core white in the output image.

    The outer fringe of the blob — where letters may be touching or bridging
    — is left completely untouched.

    Parameters
    ----------
    image             : numpy array (binary or grayscale uint8)
    min_area          : components smaller than this are always kept
    min_solidity      : compactness threshold 0–1; raise to be more conservative
    max_aspect_ratio  : elongated components (text) are always kept
    erosion_ratio     : fraction of sqrt(area) used as erosion kernel radius;
                        larger value → smaller core removed (safer)

    Returns
    -------
    cleaned binary image (uint8, same shape as input)
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Work on inverted image (blobs = white foreground on black)
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    # Start with the original binary; we will only paint white into it
    result = binary.copy()

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]

        # ── Gate 1: small → normal character → skip ───────────────────────
        if area <= min_area:
            continue

        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)

        # ── Gate 2: elongated → text stroke / border → skip ──────────────
        if aspect > max_aspect_ratio:
            continue

        # ── Gate 3: low solidity → complex text shape → skip ─────────────
        component_mask = np.uint8(labels == lbl) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        solidity  = float(area) / hull_area if hull_area > 0 else 0.0

        if solidity < min_solidity:
            continue

        # ── Confirmed blob: erode inward to get safe inner core ───────────
        # Kernel radius = erosion_ratio × sqrt(area), minimum 3 px
        k_radius = max(3, int(erosion_ratio * (area ** 0.5)))
        k_size   = 2 * k_radius + 1
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size)
        )
        core = cv2.erode(component_mask, kernel, iterations=1)

        # Paint the core white (255) in the output (white = background)
        result[core == 255] = 255

    return result


def remove_small_noise(image, max_area=20):
    """
    Remove very small connected components (scanning speckles / dust).

    Parameters
    ----------
    image    : numpy array — binary or grayscale (uint8)
    max_area : int         — components with area < max_area are removed

    Returns
    -------
    cleaned binary image (uint8, same shape as input)
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    keep_mask = np.zeros_like(inverted)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= max_area:
            keep_mask[labels == lbl] = 255

    return cv2.bitwise_not(keep_mask)


def morph_open(image, kernel_size=3):
    """
    Morphological opening with an elliptical structuring element.

    Breaks thin connections between text and blob artifacts without
    eroding character strokes as aggressively as a rectangular kernel.

    Parameters
    ----------
    image       : numpy array — binary or grayscale (uint8)
    kernel_size : int         — size of the elliptical kernel

    Returns
    -------
    opened image (uint8, same shape as input)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


OP_REGISTRY = {
    "grayscale":    convert_to_grayscale,
    "deskew":       correct_skew,
    "normalize":    normalize_image,
    "denoise":      denoise_image,
    "contrast":     enhance_contrast,
    "binarize":     binarize_image,
    "morph":        morphological_operations,
    "remove_blobs": remove_large_blobs,
    "remove_noise": remove_small_noise,
    "morph_open":   morph_open,
}

def apply_operation(image, op_name, params=None):
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Unknown operation: {op_name}")

    func = OP_REGISTRY[op_name]

    if params:
        return func(image, **params)
    return func(image)

def run_pipeline(image, pipeline):
    result = image

    for step in pipeline:
        op = step["op"]
        params = step.get("params", {})
        result = apply_operation(result, op, params)

    return result

def expand_pages(spec):
    pages = set()

    for item in spec:
        if isinstance(item, tuple):
            pages.update(range(item[0], item[1]+1))
        else:
            pages.add(item)

    return pages

def compare_images(original_path, processed_path, figsize=(12,6)):

    img1 = cv2.imread(original_path)
    img2 = cv2.imread(processed_path)

    if img1 is None or img2 is None:
        print("Error loading images")
        return

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("Processed")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def process_image_folder(
    input_dir,
    output_dir,
    default_pipeline,
    page_pipelines=None,
    ext=".png"
):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # ---------------------------------
    # Create book-specific output folder
    # ---------------------------------
    book_name = input_dir.name
    book_output_dir = output_dir / book_name
    book_output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------
    # Expand page rules
    # ---------------------------------
    page_map = {}

    if page_pipelines:
        for spec, pipeline in page_pipelines.items():
            pages = expand_pages([spec] if not isinstance(spec, list) else spec)

            for p in pages:
                page_map[p] = pipeline

    # ---------------------------------
    # Get files
    # ---------------------------------
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(ext)])

    # ---------------------------------
    # Process pages with progress bar
    # ---------------------------------
    for idx, fname in enumerate(
        tqdm(files, desc=f"Processing {book_name}", unit="page"),
        start=1
    ):

        path = input_dir / fname
        image = cv2.imread(str(path))

        if image is None:
            print("Skip:", fname)
            continue

        pipeline = page_map.get(idx, default_pipeline)

        processed = run_pipeline(image, pipeline)

        out_path = book_output_dir / fname
        cv2.imwrite(str(out_path), processed)

