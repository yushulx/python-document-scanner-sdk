import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from app import DocumentScannerApp, ProcessFileWorker, np_to_qpixmap
from scanner import DCVScanner, load_image, save_image, create_default_quad_points

TEST_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document.png")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")


def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_app_initialization():
    print("Testing app initialization...")
    app = QApplication.instance() or QApplication(sys.argv)
    window = DocumentScannerApp()
    assert window is not None
    assert window.stack is not None
    print(f"  Window size: {window.size().width()}x{window.size().height()}")
    print(f"  Initial screen: {window.stack.currentIndex()}")
    window.close()


def test_process_file_worker():
    print("\nTesting process file worker...")
    scanner = DCVScanner()
    scanner.init()

    result = {"done": False, "image": None, "quad": None, "normalized": None}

    def callback(image, quad, normalized):
        result["done"] = True
        result["image"] = image
        result["quad"] = quad
        result["normalized"] = normalized

    worker = ProcessFileWorker(scanner, TEST_INPUT)
    worker.signals.result.connect(lambda args: callback(*args))
    worker.run()

    assert result["done"], "Worker did not finish"
    assert result["image"] is not None, "Failed to load image"
    assert result["quad"] is not None, "Failed to detect document"
    assert result["normalized"] is not None, "Failed to normalize document"
    print(f"  Image shape: {result['image'].shape}")
    print(f"  Quad: {[(p.x, p.y) for p in result['quad']]}")
    print(f"  Normalized shape: {result['normalized'].shape}")

    output_path = os.path.join(OUTPUT_DIR, "app_normalized.png")
    assert save_image(result["normalized"], output_path)
    print(f"  Saved: {output_path}")


def test_page_management():
    print("\nTesting page management...")
    app = QApplication.instance() or QApplication(sys.argv)
    window = DocumentScannerApp()

    # Add a page manually
    img = load_image(TEST_INPUT)
    scanner = DCVScanner()
    scanner.init()
    quad = scanner.detect_document(img)
    normalized = scanner.normalize_document(img, quad)

    window.pages.append(type("Page", (), {
        "base_image": normalized,
        "original_image": img,
        "quad_points": quad,
        "filter_mode": "color",
    })())

    window._update_thumbnail_bar()
    window._render_result()

    assert len(window.pages) == 1
    assert window.page_indicator.text() == "1 / 1"
    print(f"  Page count: {len(window.pages)}")
    print(f"  Page indicator: {window.page_indicator.text()}")

    # Test filter
    window._on_filter("grayscale")
    assert window.pages[0].filter_mode == "grayscale"
    print("  Grayscale filter applied")

    # Test rotate
    window._on_rotate()
    assert window.pages[0].base_image.shape[0] == normalized.shape[1]
    print(f"  Rotated shape: {window.pages[0].base_image.shape}")

    window.close()


def test_edit_widget_scaling():
    print("\nTesting edit widget scaling...")
    app = QApplication.instance() or QApplication(sys.argv)
    window = DocumentScannerApp()

    img = load_image(TEST_INPUT)
    h, w = img.shape[:2]
    window.edit_widget.set_image(img)
    window.edit_widget.set_quad_points(create_default_quad_points(w, h))

    # Portrait image in a wide widget: should fit full height with side bars.
    window.edit_widget.resize(800, 400)
    img_rect = window.edit_widget.get_image_rect()
    widget_size = window.edit_widget.size()
    assert img_rect.width() <= widget_size.width() + 1
    assert img_rect.height() <= widget_size.height() + 1
    assert img_rect.height() == widget_size.height()
    expected_aspect = w / h
    actual_aspect = img_rect.width() / img_rect.height()
    assert abs(expected_aspect - actual_aspect) < 0.02, "Aspect ratio not preserved"
    print(f"  Wide widget image rect: {img_rect.width()}x{img_rect.height()}")

    # Portrait image in a tall widget: should fit full width with top/bottom bars.
    window.edit_widget.resize(400, 800)
    img_rect = window.edit_widget.get_image_rect()
    widget_size = window.edit_widget.size()
    assert img_rect.width() <= widget_size.width() + 1
    assert img_rect.height() <= widget_size.height() + 1
    assert img_rect.width() == widget_size.width()
    actual_aspect = img_rect.width() / img_rect.height()
    assert abs(expected_aspect - actual_aspect) < 0.02, "Aspect ratio not preserved"
    print(f"  Tall widget image rect: {img_rect.width()}x{img_rect.height()}")

    window.close()


def main():
    setup()
    test_app_initialization()
    test_process_file_worker()
    test_page_management()
    test_edit_widget_scaling()
    print("\nAll app tests passed!")


if __name__ == "__main__":
    main()
