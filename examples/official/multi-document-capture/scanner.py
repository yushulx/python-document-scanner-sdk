import os
import math
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from dynamsoft_capture_vision_bundle import (
    CaptureVisionRouter,
    LicenseManager,
    EnumErrorCode,
    EnumPresetTemplate,
    EnumImagePixelFormat,
    ImageData,
    ImageIO,
    ImageProcessor,
    Quadrilateral,
    Point,
)


DEFAULT_LICENSE_KEY = (
    "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6"
    "IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ=="
)

DETECT_TEMPLATE = "DetectDocumentBoundaries_Default"
NORMALIZE_TEMPLATE = "NormalizeDocument_Default"


@dataclass
class QuadPoint:
    x: float
    y: float


@dataclass
class Page:
    """Represents a captured/normalized page."""
    base_image: np.ndarray  # normalized/cropped image (RGB)
    original_image: Optional[np.ndarray] = None  # original frame before normalization (RGB)
    quad_points: Optional[List[QuadPoint]] = None
    filter_mode: str = "color"


def np_to_image_data(image: np.ndarray) -> ImageData:
    """Convert RGB or grayscale numpy array to DCV ImageData."""
    if image.ndim == 2:
        h, w = image.shape
        stride = image.strides[0]
        return ImageData(image.tobytes(), w, h, stride, EnumImagePixelFormat.IPF_GRAYSCALED)
    if image.shape[2] == 3:
        h, w = image.shape[:2]
        stride = image.strides[0]
        return ImageData(image.tobytes(), w, h, stride, EnumImagePixelFormat.IPF_RGB_888)
    if image.shape[2] == 4:
        h, w = image.shape[:2]
        stride = image.strides[0]
        return ImageData(image.tobytes(), w, h, stride, EnumImagePixelFormat.IPF_ARGB_8888)
    # Fallback: assume BGR and convert
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    stride = rgb.strides[0]
    return ImageData(rgb.tobytes(), w, h, stride, EnumImagePixelFormat.IPF_RGB_888)


def image_data_to_np(image_data: ImageData) -> np.ndarray:
    """Convert DCV ImageData to RGB numpy array."""
    fmt = image_data.get_image_pixel_format()
    w = image_data.get_width()
    h = image_data.get_height()
    stride = image_data.get_stride()
    buf = image_data.get_bytes()
    if fmt == EnumImagePixelFormat.IPF_GRAYSCALED:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        gray = arr[:, :w]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if fmt == EnumImagePixelFormat.IPF_BINARY_8:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        gray = arr[:, :w]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if fmt == EnumImagePixelFormat.IPF_BINARY_8_INVERTED:
        # DCV returns inverted 8-bit binary (0=white, 255=black); flip it back.
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        gray = 255 - arr[:, :w]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if fmt in (EnumImagePixelFormat.IPF_BINARY, EnumImagePixelFormat.IPF_BINARYINVERTED):
        # 1-bit packed binary: unpack bits and scale to 0/255.
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        bits = np.unpackbits(arr, axis=1)[:, :w] * 255
        if fmt == EnumImagePixelFormat.IPF_BINARYINVERTED:
            bits = 255 - bits
        return cv2.cvtColor(bits.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if fmt == EnumImagePixelFormat.IPF_RGB_888:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        arr = arr[:, :w * 3]
        return arr.reshape((h, w, 3))
    if fmt == EnumImagePixelFormat.IPF_BGR_888:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        arr = arr[:, :w * 3]
        bgr = arr.reshape((h, w, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == EnumImagePixelFormat.IPF_ARGB_8888:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
        arr = arr[:, :w * 4]
        return cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_RGBA2RGB)
    # Fallback: assume RGB
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
    arr = arr[:, :w * 3]
    return arr.reshape((h, w, 3))


class DCVScanner:
    def __init__(self, license_key: str = DEFAULT_LICENSE_KEY):
        self.license_key = license_key
        self.cvr: Optional[CaptureVisionRouter] = None
        self._initialized = False

    def init(self) -> Tuple[int, str]:
        if self._initialized:
            return EnumErrorCode.EC_OK, "OK"
        ec, msg = LicenseManager.init_license(self.license_key)
        if ec != EnumErrorCode.EC_OK:
            return ec, msg
        self.cvr = CaptureVisionRouter()
        self._initialized = True
        return EnumErrorCode.EC_OK, "OK"

    def detect_document(self, image: np.ndarray) -> Optional[List[QuadPoint]]:
        """Detect document boundaries in a RGB numpy image. Returns quad points or None."""
        if not self._initialized:
            self.init()
        img_data = np_to_image_data(image)
        result = self.cvr.capture(img_data, DETECT_TEMPLATE)
        if result.get_error_code() != EnumErrorCode.EC_OK:
            return None
        processed = result.get_processed_document_result()
        if not processed:
            return None
        quads = processed.get_detected_quad_result_items()
        if not quads:
            return None
        best = max(quads, key=lambda q: q.get_confidence_as_document_boundary())
        loc = best.get_location()
        return [QuadPoint(p.x, p.y) for p in loc.points]

    def detect_file(self, file_path: str) -> Optional[List[QuadPoint]]:
        if not self._initialized:
            self.init()
        result = self.cvr.capture(file_path, DETECT_TEMPLATE)
        if result.get_error_code() != EnumErrorCode.EC_OK:
            return None
        processed = result.get_processed_document_result()
        if not processed:
            return None
        quads = processed.get_detected_quad_result_items()
        if not quads:
            return None
        best = max(quads, key=lambda q: q.get_confidence_as_document_boundary())
        loc = best.get_location()
        return [QuadPoint(p.x, p.y) for p in loc.points]

    def normalize_document(self, image: np.ndarray, quad_points: Optional[List[QuadPoint]] = None) -> Optional[np.ndarray]:
        """Normalize document using provided quad points. If None, use full image."""
        if not self._initialized:
            self.init()
        img_data = np_to_image_data(image)
        template_name = NORMALIZE_TEMPLATE

        if quad_points and len(quad_points) == 4:
            ec, msg, settings = self.cvr.get_simplified_settings(template_name)
            if ec == EnumErrorCode.EC_OK:
                quad = Quadrilateral()
                quad.points = [Point(int(round(p.x)), int(round(p.y))) for p in quad_points]
                settings.roi = quad
                settings.roi_measured_in_percentage = 0
                ec2, msg2 = self.cvr.update_settings(template_name, settings)
                if ec2 != EnumErrorCode.EC_OK:
                    print(f"update_settings warning: {msg2}")

        result = self.cvr.capture(img_data, template_name)
        if result.get_error_code() != EnumErrorCode.EC_OK:
            return None
        processed = result.get_processed_document_result()
        if not processed:
            return None
        items = processed.get_enhanced_image_result_items()
        if not items:
            return None
        item = items[0]
        img_data = item.get_image_data()
        if img_data is None:
            return None
        return image_data_to_np(img_data)

    def normalize_file(self, file_path: str, quad_points: Optional[List[QuadPoint]] = None) -> Optional[np.ndarray]:
        if not self._initialized:
            self.init()
        template_name = NORMALIZE_TEMPLATE
        if quad_points and len(quad_points) == 4:
            ec, msg, settings = self.cvr.get_simplified_settings(template_name)
            if ec == EnumErrorCode.EC_OK:
                quad = Quadrilateral()
                quad.points = [Point(int(round(p.x)), int(round(p.y))) for p in quad_points]
                settings.roi = quad
                settings.roi_measured_in_percentage = 0
                self.cvr.update_settings(template_name, settings)
        result = self.cvr.capture(file_path, template_name)
        if result.get_error_code() != EnumErrorCode.EC_OK:
            return None
        processed = result.get_processed_document_result()
        if not processed:
            return None
        items = processed.get_enhanced_image_result_items()
        if not items:
            return None
        img_data = items[0].get_image_data()
        if img_data is None:
            return None
        return image_data_to_np(img_data)


# ── Image processing helpers ──


def copy_image(image: np.ndarray) -> np.ndarray:
    return image.copy()


def create_default_quad_points(width: int, height: int, inset_ratio: float = 0.15) -> List[QuadPoint]:
    max_inset_x = max(1, (width - 2) // 2)
    max_inset_y = max(1, (height - 2) // 2)
    inset_x = min(max_inset_x, max(1, min(150, round(width * inset_ratio))))
    inset_y = min(max_inset_y, max(1, min(150, round(height * inset_ratio))))
    right = width - inset_x - 1
    bottom = height - inset_y - 1
    return [
        QuadPoint(inset_x, inset_y),
        QuadPoint(right, inset_y),
        QuadPoint(right, bottom),
        QuadPoint(inset_x, bottom),
    ]


def points_to_bounding_box(points: List[QuadPoint]) -> Dict[str, float]:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}


def compute_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    if union <= 0:
        return 0.0
    return inter / union


def polygon_area(points: List[QuadPoint]) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        total += points[i].x * points[j].y - points[j].x * points[i].y
    return abs(total) * 0.5


def is_quad_stable(current: List[QuadPoint], previous: List[QuadPoint], iou_threshold: float, area_delta_threshold: float) -> bool:
    if not current or not previous or len(current) != 4 or len(previous) != 4:
        return False
    box_a = points_to_bounding_box(current)
    box_b = points_to_bounding_box(previous)
    iou = compute_iou(box_a, box_b)
    area_a = polygon_area(current)
    area_b = polygon_area(previous)
    area_delta = 1.0 if area_b == 0 else abs(area_a - area_b) / area_b
    return iou >= iou_threshold and area_delta <= area_delta_threshold


def apply_filter(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "color":
        return image.copy()

    processor = ImageProcessor()
    img_data = np_to_image_data(image)

    if mode == "grayscale":
        result_data = processor.convert_to_gray(img_data)
        return image_data_to_np(result_data)

    if mode == "binary":
        # Use global threshold 140 to match previous OpenCV behavior.
        # invert=True yields black text on white background.
        result_data = processor.convert_to_binary_global(img_data, threshold=140, invert=True)
        return image_data_to_np(result_data)

    return image.copy()


def rotate_image_90(image: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees clockwise."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    if image.shape[1] == target_width:
        return image.copy()
    h, w = image.shape[:2]
    new_h = max(1, round((h / w) * target_width))
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA if target_width < w else cv2.INTER_LANCZOS4)


def estimate_vertical_overlap(prev_img: np.ndarray, curr_img: np.ndarray) -> int:
    """Estimate vertical overlap between two images using OpenCV template matching."""
    WORK_WIDTH = 460
    MIN_CONFIDENCE = 0.42
    BAND_FRACTIONS = [0.02, 0.05, 0.09, 0.14, 0.20]

    def make_gray_work(img):
        scale = min(1.0, WORK_WIDTH / img.shape[1])
        w = max(1, round(img.shape[1] * scale))
        h = max(1, round(img.shape[0] * scale))
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred, scale, w, h

    prev_gray, prev_scale, prev_w, prev_h = make_gray_work(prev_img)
    curr_gray, curr_scale, curr_w, curr_h = make_gray_work(curr_img)

    if curr_w != prev_w:
        curr_gray = cv2.resize(curr_gray, (prev_w, curr_h), interpolation=cv2.INTER_AREA)
        curr_w = prev_w

    band_h = max(16, round(curr_h * 0.05))
    col_start = round(prev_w * 0.12)
    col_end = round(prev_w * 0.88)
    col_w = max(8, col_end - col_start)
    margin = round(curr_h * 0.015)
    search_top = round(prev_h * 0.30)
    search_h = prev_h - search_top

    if search_h <= band_h + 2 or col_w >= prev_w:
        return 0

    search_region = prev_gray[search_top:search_top + search_h, :]

    best = None
    for fraction in BAND_FRACTIONS:
        ty = margin + round(curr_h * fraction)
        if ty + band_h >= curr_h:
            continue
        templ = curr_gray[ty:ty + band_h, col_start:col_start + col_w]
        sd = templ.std()
        if sd < 8:
            continue
        result = cv2.matchTemplate(search_region, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        matched_top = search_top + max_loc[1]
        overlap_scaled = ty + (prev_h - matched_top)
        if overlap_scaled <= 0:
            continue
        if best is None or max_val > best["confidence"]:
            best = {"confidence": max_val, "overlap_scaled": overlap_scaled, "scale": prev_scale}

    if best is None or best["confidence"] < MIN_CONFIDENCE:
        return 0

    overlap = round(best["overlap_scaled"] / best["scale"])
    return max(0, min(prev_img.shape[0] - 1, curr_img.shape[0] - 1, overlap))


def build_stitched_image(images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    """Stitch images vertically. Returns stitched image and number of matched segments."""
    if not images:
        return None, 0

    widths = [img.shape[1] for img in images]
    target_width = int(sorted(widths)[len(widths) // 2])
    resized = [resize_to_width(img, target_width) for img in images]

    overlaps = []
    matched = 0
    total_height = resized[0].shape[0]
    for i in range(1, len(resized)):
        overlap = estimate_vertical_overlap(resized[i - 1], resized[i])
        overlaps.append(overlap)
        if overlap > 0:
            matched += 1
        total_height += resized[i].shape[0] - overlap

    out = np.full((total_height, target_width, 3), 255, dtype=np.uint8)
    y = 0
    out[0:resized[0].shape[0], :] = resized[0]
    y += resized[0].shape[0]

    for i in range(1, len(resized)):
        overlap = overlaps[i - 1]
        visible_h = resized[i].shape[0] - overlap
        if visible_h <= 0:
            continue
        visible = resized[i][overlap:, :]
        out[y:y + visible_h, :] = visible
        y += visible_h

    return out, matched


def save_image(image: np.ndarray, path: str) -> bool:
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, rgb)
        return True
    except Exception as e:
        print(f"save_image failed: {e}")
        return False


def load_image(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"load_image failed: {e}")
        return None


def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))
