import argparse
import os
import docscanner
from docscanner import *
import sys
import numpy as np
import cv2
import time

g_results = None
g_normalized_images = []
index = 0


def callback(results):
    global g_results
    g_results = results


def showNormalizedImage(name, normalized_image):
    cv2.imshow(name, normalized_image)


def process_file(filename, scanner):
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: Could not read image file: {filename}")
        return
    results = scanner.detect(image)
    normalized_image = None
    for result in results:
        x1 = result.x1
        y1 = result.y1
        x2 = result.x2
        y2 = result.y2
        x3 = result.x3
        y3 = result.y3
        x4 = result.x4
        y4 = result.y4

        normalized_image = scanner.normalize(result, EnumImageColourMode.ICM_COLOUR)

        cv2.drawContours(
            image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

    cv2.putText(image, 'Press "ESC" to exit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Document Image', image)
    cv2.waitKey(0)

    if normalized_image is not None:
        cv2.imwrite(str(time.time()) + '.png', normalized_image)
        print('Image saved')
    else:
        print('No document found')


def process_video(scanner, camera_index=0):
    global g_normalized_images, index
    scanner.addAsyncListener(callback)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        scanner.clearAsyncListener()
        return

    try:
        while True:
            ret, image = cap.read()
            if not ret or image is None:
                print("Error: Cannot read frame from camera")
                break

            ch = cv2.waitKey(1)
            if ch == 27:
                break
            elif ch == ord('n'):  # normalize image
                if g_results != None:

                    if len(g_results) > 0:
                        for result in g_results:
                            x1 = result.x1
                            y1 = result.y1
                            x2 = result.x2
                            y2 = result.y2
                            x3 = result.x3
                            y3 = result.y3
                            x4 = result.x4
                            y4 = result.y4

                            if result.normalized_image is not None:
                                g_normalized_images.append(
                                    (str(index), result.normalized_image))
                                showNormalizedImage(str(index), result.normalized_image)
                                index += 1
                    else:
                        print('No document found')
            elif ch == ord('s'):  # save image
                if len(g_normalized_images) > 0:
                    for data in g_normalized_images:
                        cv2.imwrite(str(time.time()) + '.png', data[1])
                        cv2.destroyWindow(data[0])
                    print('Images saved')

                    g_normalized_images = []
                    index = 0
                else:
                    print('No image to save')

            scanner.detectMatAsync(image)

            if g_results != None:
                for result in g_results:
                    x1 = result.x1
                    y1 = result.y1
                    x2 = result.x2
                    y2 = result.y2
                    x3 = result.x3
                    y3 = result.y3
                    x4 = result.x4
                    y4 = result.y4

                    contour = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)
                    cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

            cv2.putText(image, '1. Press "n" to normalize image',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, '2. Press "s" to save image', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, '3. Press "ESC" to exit', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Document Scanner', image)
    finally:
        scanner.clearAsyncListener()
        cap.release()
        cv2.destroyAllWindows()
def scandocument():
    """
    Command-line script for scanning documents from a given image or camera video stream.
    """
    parser = argparse.ArgumentParser(
        description='Scan documents from an image file or camera')
    parser.add_argument('-f', '--file', help='Path to the image file')
    parser.add_argument('-c', '--camera', default=None,
                        type=int, help='Camera index to open (for example 0)')
    parser.add_argument('-l', '--license', default='',
                        type=str, help='Set a valid license key')
    args = parser.parse_args()
    # print(args)
    try:
        filename = args.file
        license = args.license
        camera = args.camera

        if filename is None and camera is None:
            parser.print_help()
            return

        # set license
        if license == '':
            license = os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
        if license == '':
            license = "DLS2eyJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSJ9"

        error_code, error_message = docscanner.initLicense(license)
        if not docscanner.isLicenseValid(error_code):
            print("License initialization failed:", error_code, error_message)
            print("Pass a valid key with -l/--license or set DYNAMSOFT_LICENSE_KEY.")
            sys.exit(1)

        # initialize document scanner
        scanner = docscanner.createInstance()

        if filename is not None:
            process_file(filename, scanner)
        elif camera is not None:
            process_video(scanner, camera)

    except Exception as err:
        print(err)
        sys.exit(1)
