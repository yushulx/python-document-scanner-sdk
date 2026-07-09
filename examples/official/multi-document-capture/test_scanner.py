import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scanner import (
    DCVScanner, Page, QuadPoint, apply_filter, rotate_image_90,
    build_stitched_image, save_image, load_image, create_default_quad_points,
    is_quad_stable, polygon_area
)

TEST_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document.png")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_detect_and_normalize():
    print("Testing detect and normalize...")
    scanner = DCVScanner()
    ec, msg = scanner.init()
    assert ec == 0, f"Init failed: {msg}"

    img = load_image(TEST_INPUT)
    assert img is not None, f"Failed to load {TEST_INPUT}"
    print(f"  Input image shape: {img.shape}")

    quad = scanner.detect_document(img)
    assert quad is not None, "Document detection failed"
    assert len(quad) == 4, "Quad should have 4 points"
    print(f"  Detected quad: {[(p.x, p.y) for p in quad]}")

    area = polygon_area(quad)
    assert area > 0, "Quad area should be positive"
    print(f"  Quad area: {area:.0f}")

    normalized = scanner.normalize_document(img, quad)
    assert normalized is not None, "Document normalization failed"
    print(f"  Normalized image shape: {normalized.shape}")

    output_path = os.path.join(OUTPUT_DIR, "normalized.png")
    assert save_image(normalized, output_path), "Failed to save normalized image"
    print(f"  Saved: {output_path}")

    # Test default normalization (no quad)
    normalized2 = scanner.normalize_document(img, None)
    assert normalized2 is not None, "Default normalization failed"
    print(f"  Default normalized shape: {normalized2.shape}")


def test_filters():
    print("\nTesting filters...")
    img = load_image(TEST_INPUT)
    assert img is not None

    for mode in ["color", "grayscale", "binary"]:
        filtered = apply_filter(img, mode)
        assert filtered.shape == img.shape, f"Filter {mode} changed dimensions"
        output_path = os.path.join(OUTPUT_DIR, f"filter_{mode}.png")
        assert save_image(filtered, output_path)
        print(f"  Saved filter {mode}: {output_path}")


def test_rotate():
    print("\nTesting rotate...")
    img = load_image(TEST_INPUT)
    rotated = rotate_image_90(img)
    assert rotated.shape[0] == img.shape[1], "Rotate width mismatch"
    assert rotated.shape[1] == img.shape[0], "Rotate height mismatch"
    output_path = os.path.join(OUTPUT_DIR, "rotated.png")
    assert save_image(rotated, output_path)
    print(f"  Saved: {output_path}")


def test_stitch():
    print("\nTesting stitch...")
    img = load_image(TEST_INPUT)
    # Use same image twice to test stitching
    stitched, matched = build_stitched_image([img, img])
    assert stitched is not None, "Stitching failed"
    print(f"  Stitched shape: {stitched.shape}, matched overlaps: {matched}")
    output_path = os.path.join(OUTPUT_DIR, "stitched.png")
    assert save_image(stitched, output_path)
    print(f"  Saved: {output_path}")


def test_quad_stability():
    print("\nTesting quad stability...")
    points = create_default_quad_points(1000, 800)
    shifted = [QuadPoint(p.x + 2, p.y + 2) for p in points]
    assert is_quad_stable(points, shifted, 0.85, 0.15), "Similar quads should be stable"

    far = [QuadPoint(p.x + 100, p.y + 100) for p in points]
    assert not is_quad_stable(points, far, 0.85, 0.15), "Far quads should not be stable"
    print("  Stability checks passed")


def main():
    setup()
    test_detect_and_normalize()
    test_filters()
    test_rotate()
    test_stitch()
    test_quad_stability()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
