"""Real-time document scanning from a camera using the async API.

Run:
    python camera_async_api.py                 # camera 0, default trial key
    python camera_async_api.py -c 1            # camera 1
    python camera_async_api.py -l <license>    # explicit license key
    set DYNAMSOFT_LICENSE_KEY=<license>        # or use an environment variable

Keys:
    n - show/save the normalized document(s) from the latest result
    s - save the normalized documents to PNG files
    ESC - quit
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import docscanner

DEFAULT_LICENSE = "DLS2eyJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSJ9"

g_results = []
g_normalized_images = []


def callback(results):
    """Receive async detection results from the SDK thread."""
    global g_results
    g_results = results or []


def show_normalized_image(name, normalized_image):
    cv2.imshow(name, normalized_image)


def process_video(scanner, camera_index=0):
    global g_normalized_images

    scanner.addAsyncListener(callback)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        scanner.clearAsyncListener()
        return

    window_index = 0
    try:
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                print("Error: Cannot read frame from camera")
                break

            ch = cv2.waitKey(1) & 0xFF
            if ch == 27:  # ESC
                break
            if ch == ord('n'):  # show normalized image
                if not g_results:
                    print("No document found")
                else:
                    g_normalized_images = []
                    for result in g_results:
                        if result.normalized_image is None:
                            continue
                        name = str(window_index)
                        g_normalized_images.append((name, result.normalized_image))
                        show_normalized_image(name, result.normalized_image)
                        window_index += 1
            elif ch == ord('s'):  # save normalized images
                if not g_normalized_images:
                    print("No image to save")
                else:
                    for name, mat in g_normalized_images:
                        out_path = f"normalized_{time.time()}.png"
                        cv2.imwrite(out_path, mat)
                        cv2.destroyWindow(name)
                        print(f"Image saved: {out_path}")
                    g_normalized_images = []
                    window_index = 0

            scanner.detectMatAsync(image)

            for result in g_results:
                points = np.array(
                    [(result.x1, result.y1), (result.x2, result.y2),
                     (result.x3, result.y3), (result.x4, result.y4)],
                    dtype=np.int32,
                )
                cv2.drawContours(image, [points], 0, (0, 255, 0), 2)

            cv2.putText(image, 'Press "n" to normalize image',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, 'Press "s" to save image',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, 'Press "ESC" to exit',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Document Scanner', image)
    finally:
        scanner.clearAsyncListener()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Realtime document scanning with the async Dynamsoft API')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('-l', '--license', default='',
                        help='Dynamsoft license key')
    args = parser.parse_args()

    license_key = args.license or os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
    if not license_key:
        license_key = DEFAULT_LICENSE

    error_code, error_message = docscanner.initLicense(license_key)
    if not docscanner.isLicenseValid(error_code):
        print("License initialization failed:", error_code, error_message)
        print("Pass a valid key with -l/--license or set DYNAMSOFT_LICENSE_KEY.")
        sys.exit(1)

    scanner = docscanner.createInstance()
    process_video(scanner, args.camera)


if __name__ == '__main__':
    main()
