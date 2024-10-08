import argparse
import docscanner
import sys
import numpy as np
import cv2
import time


def showNormalizedImage(name, normalized_image):
    mat = docscanner.convertNormalizedImage2Mat(normalized_image)
    cv2.imshow(name, mat)
    return mat


def process_file(filename, scanner):
    image = cv2.imread(filename)
    results = scanner.detectMat(image)
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

        normalized_image = scanner.normalizeBuffer(
            image, x1, y1, x2, y2, x3, y3, x4, y4)
        showNormalizedImage("Normalized Image", normalized_image)
        cv2.drawContours(
            image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)

    cv2.putText(image, 'Press "ESC" to exit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Document Image', image)
    cv2.waitKey(0)

    if normalized_image is not None:
        normalized_image.save(str(time.time()) + '.png')
        print('Image saved')
        normalized_image.recycle()
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
        ret = scanner.setParameters(docscanner.Templates.color)

        if filename is not None:
            process_file(filename, scanner)

    except Exception as err:
        print(err)
        sys.exit(1)


scandocument()
