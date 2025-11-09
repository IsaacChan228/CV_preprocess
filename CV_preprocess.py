import os
import sys
import glob
from PIL import Image

######## Global configuration ########
side_length = 640
jpeg_quality = 95

# Input / output folders
INPUT_IMAGES_DIR = 'input/images'
INPUT_LABELS_DIR = 'input/labels'
OUTPUT_IMAGES_DIR = 'output/images'
OUTPUT_LABELS_DIR = 'output/labels'

# Allowed classes for filtering.
# If None -> allow any class.
# Otherwise provide a list of integer class ids.
# Example: ALLOWED_CLASSES = [0, 1, 2]
ALLOWED_CLASSES = [4]

# Target class
# replace the class in the label to this target class id
TARGET_CLASS = 27

# Target output name
# replace the name of the image and label with this variable
TARGET_NAME = "EGGS"

######## Code ########

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_image_file(path):
    lower = path.lower()
    return lower.endswith((
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'
    ))


def main():
    # read the input/output directories
    in_images = os.path.abspath(INPUT_IMAGES_DIR)
    in_labels = os.path.abspath(INPUT_LABELS_DIR)
    out_images = os.path.abspath(OUTPUT_IMAGES_DIR)
    out_labels = os.path.abspath(OUTPUT_LABELS_DIR)

    if not os.path.isdir(in_images):
        print(f"Input images directory does not exist: {in_images}")
        sys.exit(1)

    if not os.path.isdir(in_labels):
        print(f"Input labels directory does not exist: {in_labels}")
        sys.exit(1)

    ensure_dir(out_images)
    ensure_dir(out_labels)

    # Collect image files
    candidates = glob.glob(os.path.join(in_images, '*'))
    files = [p for p in candidates if os.path.isfile(p) and is_image_file(p)]

    if not files:
        print(f"No image files found in {in_images}")
        return

    # prepare log file at repository root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(repo_root, 'log.txt')
    with open(log_path, 'w') as log_fh:
        processed_idx = 1
        for img_path in files:
            try:
                ok, reason = process_image_and_label(img_path, in_labels, out_images, out_labels,
                                                     size=side_length, quality=jpeg_quality,
                                                     out_index=processed_idx, target_name=TARGET_NAME)
                if not ok:
                    # write skip reason to log
                    log_fh.write(f"{os.path.basename(img_path)}\tSKIPPED\t{reason}\n")
                else:
                    processed_idx += 1
            except Exception as e:
                # log unexpected failure
                log_fh.write(f"{os.path.basename(img_path)}\tERROR\t{e}\n")
    print(f"Processing completed. Log written to {log_path}")


def read_yolo_label(label_path):
    """Read YOLO format label file. Returns list of (cls:int, x,y,w,h) normalized floats."""
    ann = []
    if not os.path.isfile(label_path):
        return ann
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            ann.append((cls, x, y, w, h))
    return ann


def write_yolo_label(label_path, ann_list):
    with open(label_path, 'w') as f:
        for cls, x, y, w, h in ann_list:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def classes_allowed(ann_list):
    if ALLOWED_CLASSES is None:
        return True
    for cls, *_ in ann_list:
        if cls not in ALLOWED_CLASSES:
            return False
    return True


def process_image_and_label(img_path, in_labels_dir, out_images_dir, out_labels_dir, size, quality, out_index, target_name):
    # Process one image+label. Returns (True, None) on success, (False, reason) on skip/error.
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(in_labels_dir, base + '.txt')
    if not os.path.isfile(label_path):
        return False, 'label not found'

    ann = read_yolo_label(label_path)
    if not ann:
        return False, 'empty label'

    if not classes_allowed(ann):
        return False, 'contains disallowed class'

    # Read image with Pillow
    try:
        pil_img = Image.open(img_path).convert('RGBA')
    except Exception as e:
        return False, f'cannot read image ({e})'

    orig_w, orig_h = pil_img.size
    side = max(orig_w, orig_h)

    # Create white square background and paste image centered
    canvas = Image.new('RGBA', (side, side), (255, 255, 255, 255))
    left = (side - orig_w) // 2
    top = (side - orig_h) // 2
    # use alpha channel as mask to composite
    mask = pil_img.split()[3] if pil_img.mode == 'RGBA' else None
    canvas.paste(pil_img, (left, top), mask)

    # Convert annotations from normalized (w,h) to pixel coordinates then shift by left/top
    ann_px = []
    for cls, x_n, y_n, w_n, h_n in ann:
        x_px = x_n * orig_w
        y_px = y_n * orig_h
        w_px = w_n * orig_w
        h_px = h_n * orig_h
        x_px_new = x_px + left
        y_px_new = y_px + top
        # normalize by side
        x_new_n = x_px_new / side
        y_new_n = y_px_new / side
        w_new_n = w_px / side
        h_new_n = h_px / side
        # replace output class with TARGET_CLASS as requested
        ann_px.append((TARGET_CLASS, x_new_n, y_new_n, w_new_n, h_new_n))

    # Resize canvas to final size (convert to RGB for saving as JPEG)
    resized = canvas.convert('RGB').resize((size, size), resample=Image.LANCZOS)

    # prepare output base name using target_name and index
    out_base = f"{target_name}_{out_index:04d}"

    # Save image and label for original orientation
    out_img_path = os.path.join(out_images_dir, out_base + '.jpg')
    out_label_path = os.path.join(out_labels_dir, out_base + '.txt')
    try:
        resized.save(out_img_path, format='JPEG', quality=quality)
    except Exception as e:
        return False, f'failed saving image: {e}'
    write_yolo_label(out_label_path, ann_px)

    # Generate rotated copies (90, 180, 270 degrees clockwise)
    ann_list = ann_px
    # 90
    ann_rot90 = [(cls, y, 1 - x, h, w) for (cls, x, y, w, h) in ann_list]
    rot90_img = resized.rotate(-90, expand=False)
    out_img_r = os.path.join(out_images_dir, out_base + '_rot90.jpg')
    out_label_r = os.path.join(out_labels_dir, out_base + '_rot90.txt')
    try:
        rot90_img.save(out_img_r, format='JPEG', quality=quality)
        write_yolo_label(out_label_r, ann_rot90)
    except Exception:
        # ignore rotated save errors but consider original processed
        pass

    # 180
    ann_rot180 = [(cls, 1 - x, 1 - y, w, h) for (cls, x, y, w, h) in ann_list]
    rot180_img = resized.rotate(180, expand=False)
    out_img_r = os.path.join(out_images_dir, out_base + '_rot180.jpg')
    out_label_r = os.path.join(out_labels_dir, out_base + '_rot180.txt')
    try:
        rot180_img.save(out_img_r, format='JPEG', quality=quality)
        write_yolo_label(out_label_r, ann_rot180)
    except Exception:
        pass

    # 270
    ann_rot270 = [(cls, 1 - y, x, h, w) for (cls, x, y, w, h) in ann_list]
    rot270_img = resized.rotate(-270, expand=False)
    out_img_r = os.path.join(out_images_dir, out_base + '_rot270.jpg')
    out_label_r = os.path.join(out_labels_dir, out_base + '_rot270.txt')
    try:
        rot270_img.save(out_img_r, format='JPEG', quality=quality)
        write_yolo_label(out_label_r, ann_rot270)
    except Exception:
        pass

    return True, None


if __name__ == '__main__':
    main()
