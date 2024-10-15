# Dynamsoft Capture Vision SDK for Document Detection
This repository contains example code for using the Dynamsoft Capture Vision SDK to detect documents in images.

https://github.com/user-attachments/assets/1d93597e-45f0-487f-a946-8f6dbcda9b07

## Prerequisites
- [Dynamsoft Capture Vision Trial License](https://www.dynamsoft.com/customer/license/trialLicense/?product=dcv&package=cross-platform)
    
    ```python
    errorCode, errorMsg = LicenseManager.init_license(
        "LICENSE-KEY")
    ```
    
- SDK Installation
 
    ```bash
    pip install -r requirements.txt
    ```

## Supported Platforms
- Windows
- Linux
- macOS
    
  
## Examples
- [camera.py](./camera.py): Detect documents from a camera video stream.
- [file.py](./file.py): Detect documents from an image file and display the results in a window.
- [orientation_detection.py](./orientation_detection.py): Normalize a document using Dynamsoft Capture Vision SDK, and detect the orientation of the document using Tesseract OCR.