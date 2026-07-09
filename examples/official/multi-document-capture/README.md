# Dynamsoft Document Scanner (Desktop)

A PySide6 desktop document scanner app built with the Dynamsoft Capture Vision SDK. It performs real-time document boundary detection and perspective normalization, with support for both USB camera and image file input.

## Features

- USB camera live preview with real-time document boundary overlay
- Auto-capture when the document boundary is stable
- Manual capture with shutter button
- Image file import
- Document normalization / perspective correction
- Color / Grayscale / Binary filters
- Page rotation
- Reorder pages via drag-and-drop
- Export as PDF, images, or stitched long image
- Quad editing with draggable corners

## Requirements

- Python 3.9+
- Windows 10/11 (camera via OpenCV DShow)
- USB camera or image files

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The app starts with a license screen. The default 1-day trial license is pre-filled. You can replace it with your own Dynamsoft Capture Vision license key.

## Project Structure

- `main.py` - Application entry point
- `app.py` - PySide6 main window and UI components
- `scanner.py` - Dynamsoft Capture Vision wrapper and image processing helpers
- `test_scanner.py` - Unit tests for detection/normalization logic
- `test_app.py` - Unit tests for the UI layer
- `requirements.txt` - Python dependencies

## Testing

```bash
python test_scanner.py
python test_app.py
```

The tests verify that `document.png` from the parent directory is correctly detected, normalized, filtered, rotated, and stitched.

## License

This example uses the default Dynamsoft 1-day trial license. For production use, obtain a valid license from [Dynamsoft](https://www.dynamsoft.com/customer/license/trialLicense/?product=dcv&package=cross-platform).
