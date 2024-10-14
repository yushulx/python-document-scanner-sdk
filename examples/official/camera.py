from dynamsoft_capture_vision_bundle import *
import cv2
import numpy as np
import queue


def convertMat2ImageData(mat):
    if len(mat.shape) == 3:
        height, width, channels = mat.shape
        pixel_format = EnumImagePixelFormat.IPF_RGB_888
    else:
        height, width = mat.shape
        channels = 1
        pixel_format = EnumImagePixelFormat.IPF_GRAYSCALED

    stride = width * channels
    imagedata = ImageData(mat.tobytes(), width, height, stride, pixel_format)
    return imagedata


class FrameFetcher(ImageSourceAdapter):
    def has_next_image_to_fetch(self) -> bool:
        return True

    def add_frame(self, imageData):
        self.add_image_to_buffer(imageData)


class MyCapturedResultReceiver(CapturedResultReceiver):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue

    def on_captured_result_received(self, captured_result):
        self.result_queue.put(captured_result)


if __name__ == '__main__':
    errorCode, errorMsg = LicenseManager.init_license(
        "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")
    if errorCode != EnumErrorCode.EC_OK and errorCode != EnumErrorCode.EC_LICENSE_CACHE_USED:
        print("License initialization failed: ErrorCode:",
              errorCode, ", ErrorString:", errorMsg)
    else:
        vc = cv2.VideoCapture(0)
        if not vc.isOpened():
            print("Error: Camera is not opened!")
            exit(1)

        cvr = CaptureVisionRouter()
        fetcher = FrameFetcher()
        cvr.set_input(fetcher)

        # Create a thread-safe queue to store captured items
        result_queue = queue.Queue()

        receiver = MyCapturedResultReceiver(result_queue)
        cvr.add_result_receiver(receiver)

        errorCode, errorMsg = cvr.start_capturing("Default")

        if errorCode != EnumErrorCode.EC_OK:
            print("error:", errorMsg)

        while True:
            ret, frame = vc.read()
            if not ret:
                print("Error: Cannot read frame!")
                break

            fetcher.add_frame(convertMat2ImageData(frame))

            # Check if there are any new captured items from the queue
            if not result_queue.empty():
                captured_result = result_queue.get_nowait()
                items = captured_result.get_items()
                for item in items:
                    location = item.get_location()
                    if item.get_type() == EnumCapturedResultItemType.CRIT_BARCODE:
                        x1 = location.points[0].x
                        y1 = location.points[0].y
                        x2 = location.points[1].x
                        y2 = location.points[1].y
                        x3 = location.points[2].x
                        y3 = location.points[2].y
                        x4 = location.points[3].x
                        y4 = location.points[3].y
                        cv2.drawContours(
                            frame, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
                    elif item.get_type() == EnumCapturedResultItemType.CRIT_NORMALIZED_IMAGE:
                        x1 = location.points[0].x
                        y1 = location.points[0].y
                        x2 = location.points[1].x
                        y2 = location.points[1].y
                        x3 = location.points[2].x
                        y3 = location.points[2].y
                        x4 = location.points[3].x
                        y4 = location.points[3].y
                        cv2.drawContours(
                            frame, [np.intp([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (255, 0, 0), 2)

                    del location

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            cv2.imshow('frame', frame)

        cvr.stop_capturing()
        vc.release()
        cv2.destroyAllWindows()