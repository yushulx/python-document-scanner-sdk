from time import sleep
import docscanner
import numpy as np
import cv2
import time

# set license
docscanner.initLicense(
    "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")

scanner = docscanner.createInstance()

ret = scanner.setParameters(docscanner.Templates.binary)
print(ret)


def showNormalizedImage(name, normalized_image):
    mat = docscanner.convertNormalizedImage2Mat(normalized_image)
    cv2.imshow(name, mat)
    return mat

# detectFile()


def test_detectFile():
    print('')
    print('Test detectFile()')
    results = scanner.detectFile("images/1.png")
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

        normalized_image = scanner.normalizeFile(
            "images/1.png", x1, y1, x2, y2, x3, y3, x4, y4)
        normalized_image.recycle()

    # assert len(results) > 0
        # showNormalizedImage("Normalized Image", normalized_image)
        # cv2.drawContours(image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)

    # cv2.imshow('Document Image', image)
    # cv2.waitKey(0)


# detectMat()
def test_detectMat():
    print('')
    print('Test detectMat()')

    image = cv2.imread("images/1.png")
    results = scanner.detectMat(image)
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

        normalized_image = scanner.normalizeBuffer(
            image, x1, y1, x2, y2, x3, y3, x4, y4)
        # showNormalizedImage("Normalized Image", normalized_image)
        normalized_image.recycle()
        cv2.drawContours(
            image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)

    # cv2.imshow('Document Image', image)
    # cv2.waitKey(0)

# detectMatAsync()


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
            cv2.drawContours(
                image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)

        # cv2.imshow('Document Image', image)
        # cv2.waitKey(0)

    image = cv2.imread("images/1.png")
    scanner.addAsyncListener(callback)
    scanner.detectMatAsync(image)


test_detectFile()
test_detectMatAsync()
sleep(3)
