import cv2
import numpy as np
import os
import sys
package_path = os.path.dirname(__file__) + '/../../'
print(package_path)
sys.path.append(package_path)
import docscanner
from docscanner import *

docscanner.initLicense(
    "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")


class Scanner(object):
    def __init__(self):
        self.scanner = docscanner.createInstance()

    def __del__(self):
        pass

    def detect_edge(self, image, enabled_transform=False):
        results = self.scanner.detect(image)
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
                image, [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)], 0, (0, 255, 0), 2)

            if enabled_transform:
                normalized_image = self.scanner.normalize(result, EnumImageColourMode.ICM_COLOUR)
            break

        return image, normalized_image
