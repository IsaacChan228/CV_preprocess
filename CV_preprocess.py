#!/usr/bin/env python3
"""
CV_preprocess - command line tool to pad images to centered square and resize to680x680, saving as JPG.

Usage:
 python CV_preprocess.py --input Input --output Output

The script prefers OpenCV (`cv2`). If not available it falls back to Pillow (`PIL`).
It also attempts to load a local `cv2` package if placed next to this script in a
`vendor/` or `opencv/` directory to reduce external dependency needs.
"""

import os
import sys
import argparse
import glob

# Try to pick up a vendored cv2 if present next to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
for vendor_name in ("vendor", "opencv"):
    vendor_dir = os.path.join(script_dir, vendor_name)
    if os.path.isdir(vendor_dir) and vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

HAVE_CV2 = False
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# Ensure numpy is available when cv2 is used, otherwise use Pillow fallback
if HAVE_CV2:
    try:
        import numpy as np
    except Exception:
        print("Error: numpy is required when using OpenCV (cv2). Install numpy or use Pillow instead.")
        sys.exit(1)
else:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        print(
            "Error: Neither OpenCV (cv2) nor Pillow (PIL) are available.\n"
            "Place OpenCV next to this script in a 'vendor' or 'opencv' folder or install one of the libraries."
        )
        sys.exit(1)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_image_file(path):
    lower = path.lower()
    return lower.endswith((
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'
    ))


def process_with_cv2(in_path, out_path, size=680, quality=95):
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image: {in_path}")

    # Normalize image to BGR or BGRA
    h, w = img.shape[:2]
    side = max(h, w)

    # Prepare white background canvas
    if img.ndim ==2:
        # grayscale -> convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas =255 * np.ones((side, side,3), dtype=img_bgr.dtype)
        top = (side - h) //2
        left = (side - w) //2
        canvas[top:top + h, left:left + w] = img_bgr
    elif img.shape[2] ==4:
        # BGRA with alpha: composite over white background
        b, g, r, a = cv2.split(img)
        alpha = a.astype(float) /255.0
        # create white background
        canvas =255 * np.ones((side, side,3), dtype=img.dtype)
        top = (side - h) //2
        left = (side - w) //2
        # Composite per-channel
        roi_b = canvas[top:top + h, left:left + w,0].astype(float)
        roi_g = canvas[top:top + h, left:left + w,1].astype(float)
        roi_r = canvas[top:top + h, left:left + w,2].astype(float)
        canvas[top:top + h, left:left + w,0] = (b.astype(float) * alpha + roi_b * (1 - alpha)).astype(img.dtype)
        canvas[top:top + h, left:left + w,1] = (g.astype(float) * alpha + roi_g * (1 - alpha)).astype(img.dtype)
        canvas[top:top + h, left:left + w,2] = (r.astype(float) * alpha + roi_r * (1 - alpha)).astype(img.dtype)
    else:
        # BGR
        canvas =255 * np.ones((side, side,3), dtype=img.dtype)
        top = (side - h) //2
        left = (side - w) //2
        canvas[top:top + h, left:left + w] = img

    # Resize to target size
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    resized = cv2.resize(canvas, (size, size), interpolation=interp)

    cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def process_with_pil(in_path, out_path, size=680, quality=95):
    img = Image.open(in_path)
    # Convert to RGBA if has alpha so we can composite on white
    if img.mode in ("RGBA", "LA"):
        img_rgba = img.convert("RGBA")
    else:
        img_rgba = img.convert("RGB")

    w, h = img_rgba.size
    side = max(w, h)

    # Create white background
    background = Image.new('RGB', (side, side), (255,255,255))
    left = (side - w) //2
    top = (side - h) //2

    if img_rgba.mode == 'RGBA':
        # paste with alpha as mask
        background.paste(img_rgba.convert('RGBA'), (left, top), img_rgba.split()[3])
    else:
        background.paste(img_rgba.convert('RGB'), (left, top))

    # Resize with high quality
    resized = background.resize((size, size), Image.LANCZOS)
    resized.save(out_path, format='JPEG', quality=quality)


def process_file(path, out_dir, size=680, quality=95):
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, base + '.jpg')
    try:
        if HAVE_CV2:
            process_with_cv2(path, out_path, size=size, quality=quality)
        else:
            process_with_pil(path, out_path, size=size, quality=quality)
        print(f"Processed: {path} -> {out_path}")
    except Exception as e:
        print(f"Failed processing {path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Pad images to centered square, resize to680x680 and save as JPG.'
    )
    parser.add_argument('--input', '-i', default='Input', help='Input directory containing images')
    parser.add_argument('--output', '-o', default='Output', help='Output directory for processed JPGs')
    parser.add_argument('--size', '-s', type=int, default=680, help='Output square size in pixels (default680)')
    parser.add_argument('--quality', '-q', type=int, default=95, help='JPEG quality1-100 (default95)')
    args = parser.parse_args()

    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)

    if not os.path.isdir(in_dir):
        print(f"Input directory does not exist: {in_dir}")
        sys.exit(1)

    ensure_dir(out_dir)

    # Collect image files
    candidates = glob.glob(os.path.join(in_dir, '*'))
    files = [p for p in candidates if os.path.isfile(p) and is_image_file(p)]

    if not files:
        print(f"No image files found in {in_dir}")
        return

    for path in files:
        process_file(path, out_dir, size=args.size, quality=args.quality)


if __name__ == '__main__':
    main()
