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
def showNormalizedImage(name, normalized_image):
    cv2.imshow(name, normalized_image)

def process_file(filename, scanner):
    image = cv2.imread(filename)
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

        if result.normalized_image is not None:
            showNormalizedImage("Normalized Image", result.normalized_image)
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


def scandocument():
    """
    Command-line script for scanning documents from a given image
    """
    parser = argparse.ArgumentParser(
        description='Scan documents from an image file')
    parser.add_argument('-f', '--file', help='Path to the image file')
    parser.add_argument('-l', '--license', default='',
                        type=str, help='Set a valid license key')
    args = parser.parse_args()
    # print(args)
    try:
        filename = args.file
        license = args.license

        if filename is None:
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

        if filename is not None:
            process_file(filename, scanner)

    except Exception as err:
        print(err)
        sys.exit(1)


scandocument()
