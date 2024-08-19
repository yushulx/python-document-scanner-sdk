import cv2
import numpy as np
import docscanner

docscanner.initLicense(
    "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")


class Scanner(object):
    def __init__(self):
        self.scanner = docscanner.createInstance()
        self.scanner.setParameters(docscanner.Templates.color)

    def __del__(self):
        pass

    def detect_edge(self, image, enabled_transform=False):
        results = self.scanner.detectMat(image)
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

            cv2.drawContours(
                image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)

            if enabled_transform:
                normalized_image = self.scanner.normalizeBuffer(
                    image, x1, y1, x2, y2, x3, y3, x4, y4)
                normalized_image = docscanner.convertNormalizedImage2Mat(
                    normalized_image)
            break

        return image, normalized_image
