import argparse
import os
import sys
package_path = os.path.dirname(__file__) + '/../../'
print(package_path)
sys.path.append(package_path)
import docscanner
from docscanner import *
import numpy as np
import cv2
import time
print(docscanner.__version__)

g_results = None
g_normalized_images = []
index = 0


def callback(results):
    global g_results
    g_results = results


def showNormalizedImage(name, normalized_image):
    cv2.imshow(name, normalized_image)


def process_video(scanner):
    global g_normalized_images, index
    scanner.addAsyncListener(callback)

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()

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

        if image is not None:
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

    scanner.clearAsyncListener()
def scandocument():
    """
    Command-line script for scanning documents from camera video stream.
    """
    parser = argparse.ArgumentParser(description='Scan documents from camera')
    parser.add_argument('-c', '--camera', default=False,
                        type=bool, help='Whether to show the image')
    parser.add_argument('-l', '--license', default='',
                        type=str, help='Set a valid license key')
    args = parser.parse_args()
    # print(args)
    try:
        license = args.license
        camera = args.camera

        if camera is False:
            parser.print_help()
            return

        # set license
        if license == '':
            docscanner.initLicense(
                "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")
        else:
            docscanner.initLicense(license)

        # initialize mrz scanner
        scanner = docscanner.createInstance()

        if camera is True:
            process_video(scanner)

    except Exception as err:
        print(err)
        sys.exit(1)


scandocument()
