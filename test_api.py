from time import sleep
import os
import docscanner
from docscanner import *
import numpy as np
import cv2
import time

print(docscanner.__version__)
# set license
license_key = os.environ.get('DYNAMSOFT_LICENSE_KEY', '')
if license_key == '':
    license_key = "DLS2eyJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSJ9"
error_code, error_message = docscanner.initLicense(license_key)
if not docscanner.isLicenseValid(error_code):
    print("License initialization failed:", error_code, error_message)
    print("Set DYNAMSOFT_LICENSE_KEY to a valid Dynamsoft license key.")
    raise SystemExit(1)

scanner = docscanner.createInstance()


def showNormalizedImage(name, mat):
    cv2.imshow(name, mat)
    return mat

# detectFile()
def test_detectFile():
    print('')
    print('Test detectFile()')
    results = scanner.detect("images/1.png")
    assert len(results) > 0
    image = cv2.imread("images/1.png")
    for result in results:
        x1 = result.x1
        y1 = result.y1
        x2 = result.x2
        y2 = result.y2
        x3 = result.x3
        y3 = result.y3
        x4 = result.x4
        y4 = result.y4

        print(x1, y1, x2, y2, x3, y3, x4, y4)

        scanner.normalize(result, EnumImageColourMode.ICM_BINARY)

        if result.normalized_image is not None:
            cv2.imwrite(str(time.time()) + '.png', result.normalized_image)

    #     if result.normalized_image is not None:
    #         showNormalizedImage("Normalized Image", result.normalized_image)
    #     cv2.drawContours(
    #         image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

    # cv2.imshow('Document Image', image)
    # cv2.waitKey(0)


# detectMat()
def test_detectMat():
    print('')
    print('Test detectMat()')

    image = cv2.imread("images/1.png")
    results = scanner.detect(image)
    assert len(results) > 0
    for result in results:
        x1 = result.x1
        y1 = result.y1
        x2 = result.x2
        y2 = result.y2
        x3 = result.x3
        y3 = result.y3
        x4 = result.x4
        y4 = result.y4
        print(x1, y1, x2, y2, x3, y3, x4, y4)
        scanner.normalize(result, EnumImageColourMode.ICM_GRAYSCALE)

        if result.normalized_image is not None:
            cv2.imwrite(str(time.time()) + '.png', result.normalized_image)
    #         showNormalizedImage("Normalized Image", result.normalized_image)
    #     cv2.drawContours(
    #         image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

    # cv2.imshow('Document Image', image)
    # cv2.waitKey(0)


def test_detectMatAsync():
    print('')
    print('Test detectMatAsync()')

    def callback(results):
        assert len(results) > 0
        for result in results:
            x1 = result.x1
            y1 = result.y1
            x2 = result.x2
            y2 = result.y2
            x3 = result.x3
            y3 = result.y3
            x4 = result.x4
            y4 = result.y4
            print(x1, y1, x2, y2, x3, y3, x4, y4)

            if result.normalized_image is not None:
                cv2.imwrite(str(time.time()) + '.png', result.normalized_image)
            # cv2.drawContours(
            # image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

            # if result.normalized_image is not None:
            #     showNormalizedImage("Normalized Image", result.normalized_image)

        # cv2.imshow('Document Image', image)
        # cv2.waitKey(0)

    image = cv2.imread("images/1.png")
    scanner.addAsyncListener(callback)
    

    for i in range(2):
        print('detectMatAsync: {}'.format(i))
        scanner.detectMatAsync(image)

        sleep(1)
    scanner.clearAsyncListener()


test_detectFile()
test_detectMat()
test_detectMatAsync()