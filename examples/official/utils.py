from dynamsoft_capture_vision_bundle import *
import numpy as np


def convertImageData2Mat(image_data):
    """Convert a Dynamsoft ImageData object to an OpenCV BGR/gray matrix.

    All pixel formats returned by the document normalizer are supported:
    packed 1-bit binary, 8-bit binary, grayscale, RGB/BGR and ARGB/ABGR.
    Stride padding is removed, so the returned array is tightly packed.
    """
    width = image_data.get_width()
    height = image_data.get_height()
    stride = image_data.get_stride()
    pixel_format = image_data.get_image_pixel_format()
    raw = np.frombuffer(image_data.get_bytes(), dtype=np.uint8)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    if stride <= 0:
        stride = len(raw) // height
    if len(raw) < stride * height:
        raise ValueError(
            f"ImageData buffer is too small: {len(raw)} bytes, "
            f"expected {stride * height} (stride={stride}, height={height})"
        )

    rows = raw[: stride * height].reshape(height, stride)

    if pixel_format in (EnumImagePixelFormat.IPF_BINARY, EnumImagePixelFormat.IPF_BINARYINVERTED):
        gray = np.unpackbits(rows, axis=1)[:, :width].astype(np.uint8) * 255
        if pixel_format == EnumImagePixelFormat.IPF_BINARYINVERTED:
            gray = 255 - gray
        return np.ascontiguousarray(gray)

    if pixel_format in (
        EnumImagePixelFormat.IPF_GRAYSCALED,
        EnumImagePixelFormat.IPF_BINARY_8,
        EnumImagePixelFormat.IPF_BINARY_8_INVERTED,
    ):
        gray = rows[:, :width]
        if pixel_format == EnumImagePixelFormat.IPF_BINARY_8_INVERTED:
            gray = 255 - gray
        return np.ascontiguousarray(gray)

    if pixel_format in (EnumImagePixelFormat.IPF_RGB_888, EnumImagePixelFormat.IPF_BGR_888):
        channels = 3
    elif pixel_format in (EnumImagePixelFormat.IPF_ARGB_8888, EnumImagePixelFormat.IPF_ABGR_8888):
        channels = 4
    else:
        raise ValueError(f"Unsupported pixel format: {pixel_format}")

    if stride < width * channels:
        raise ValueError(
            f"Invalid stride {stride} for a {width}x{height} image with "
            f"{channels} channels"
        )

    mat = rows[:, : width * channels].reshape(height, width, channels)
    if pixel_format == EnumImagePixelFormat.IPF_RGB_888:
        return np.ascontiguousarray(mat[:, :, ::-1])  # RGB -> BGR
    if pixel_format == EnumImagePixelFormat.IPF_BGR_888:
        return np.ascontiguousarray(mat)
    if pixel_format == EnumImagePixelFormat.IPF_ARGB_8888:
        return np.ascontiguousarray(mat[:, :, 3:0:-1])  # ARGB -> BGR
    # IPF_ABGR_8888 (A, B, G, R) -> BGR
    return np.ascontiguousarray(mat[:, :, 1:4])


def convertMat2ImageData(mat):
    """Convert an OpenCV (RGB or grayscale) matrix to ImageData."""
    if not isinstance(mat, np.ndarray):
        raise TypeError("mat must be a numpy.ndarray")

    mat = np.ascontiguousarray(mat)
    if mat.ndim == 3 and mat.shape[2] == 1:
        mat = mat[:, :, 0]

    if mat.ndim == 3 and mat.shape[2] == 3:
        height, width = mat.shape[:2]
        pixel_format = EnumImagePixelFormat.IPF_RGB_888
    elif mat.ndim == 2:
        height, width = mat.shape
        pixel_format = EnumImagePixelFormat.IPF_GRAYSCALED
    else:
        raise ValueError(
            "Unsupported image shape %r; expected (H, W), (H, W, 1) or (H, W, 3)"
            % (mat.shape,)
        )

    return ImageData(mat.tobytes(), width, height, mat.strides[0], pixel_format)
