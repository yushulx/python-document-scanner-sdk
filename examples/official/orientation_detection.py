from dynamsoft_capture_vision_bundle import *
import os
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def convertNormalizedImage2Mat(normalized_image):
    ba = bytearray(normalized_image.get_bytes())
    width = normalized_image.get_width()
    height = normalized_image.get_height()

    channels = 3
    if normalized_image.get_image_pixel_format() == EnumImagePixelFormat.IPF_BINARY:
        channels = 1
        all = []
        skip = normalized_image.stride * 8 - width

        index = 0
        n = 1
        for byte in ba:

            byteCount = 7
            while byteCount >= 0:
                b = (byte & (1 << byteCount)) >> byteCount

                if index < normalized_image.stride * 8 * n - skip:
                    if b == 1:
                        all.append(255)
                    else:
                        all.append(0)

                byteCount -= 1
                index += 1

            if index == normalized_image.stride * 8 * n:
                n += 1

        mat = np.array(all, dtype=np.uint8).reshape(height, width, channels)
        return mat

    elif normalized_image.get_image_pixel_format() == EnumImagePixelFormat.IPF_GRAYSCALED:
        channels = 1

    mat = np.array(ba, dtype=np.uint8).reshape(height, width, channels)

    return mat


if __name__ == '__main__':
    errorCode, errorMsg = LicenseManager.init_license(
        "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")
    if errorCode != EnumErrorCode.EC_OK and errorCode != EnumErrorCode.EC_LICENSE_CACHE_USED:
        print("License initialization failed: ErrorCode:",
              errorCode, ", ErrorString:", errorMsg)
    else:
        cvr = CaptureVisionRouter()
        while (True):
            image_path = input(
                ">> Input your image full path:\n"
                ">> 'Enter' for sample image or 'Q'/'q' to quit\n"
            ).strip('\'"')

            if image_path.lower() == "q":
                sys.exit(0)

            if not os.path.exists(image_path):
                print("The image path does not exist.")
                continue
            result = cvr.capture(
                image_path, EnumPresetTemplate.PT_DETECT_AND_NORMALIZE_DOCUMENT.value)
            if result.get_error_code() != EnumErrorCode.EC_OK:
                print("Error:", result.get_error_code(),
                      result.get_error_string())
            normalized_images_result = result.get_normalized_images_result()
            if normalized_images_result is None or len(normalized_images_result.get_items()) == 0:
                print("No normalized documents.")
            else:
                items = normalized_images_result.get_items()
                print("Normalized", len(items), "documents.")
                for index, item in enumerate(normalized_images_result.get_items()):
                    out_path = "normalizedResult_" + str(index) + ".png"
                    image_manager = ImageManager()
                    image = item.get_image_data()
                    if image != None:

                        mat = convertNormalizedImage2Mat(image)
                        # Use Tesseract to determine the character orientation in the warped image
                        osd_data = pytesseract.image_to_osd(
                            mat, output_type=Output.DICT)
                        rotation_angle = osd_data['rotate']

                        print(
                            f"Detected Character Orientation: {rotation_angle} degrees")

                        # Draw the detected rotation angle on the original image
                        cv_image = cv2.imread(image_path)

                        location = item.get_location()
                        x1 = location.points[0].x
                        y1 = location.points[0].y
                        x2 = location.points[1].x
                        y2 = location.points[1].y
                        x3 = location.points[2].x
                        y3 = location.points[2].y
                        x4 = location.points[3].x
                        y4 = location.points[3].y

                        cv2.drawContours(
                            cv_image, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
                        cv2.putText(cv_image, f"Rotation: {rotation_angle} degrees", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow(
                            "Original Image with Detected Border and Rotation Angle", cv_image)
                        cv2.imshow("Normalized Image", mat)
                        cv2.waitKey(0)

                        errorCode, errorMsg = image_manager.save_to_file(
                            image, out_path)
                        if errorCode == 0:
                            print("Document " + str(index) +
                                  " file: " + out_path)
    input("Press Enter to quit...")
