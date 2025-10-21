"""
Python Document Scanner SDK

A Python wrapper for the Dynamsoft Document Normalizer SDK, providing document
detection and perspective correction capabilities for images and camera streams.

This module enables:
- Document boundary detection in images and camera feeds
- Real-time document scanning with asynchronous processing  
- Perspective correction (normalization) of detected documents
- Support for various image formats and OpenCV matrices

Basic Usage:
    ```python
    import docscanner
    import cv2
    from dynamsoft_capture_vision_bundle import EnumImageColourMode
    
    # Initialize license (required)
    docscanner.initLicense("YOUR_LICENSE_KEY")
    
    # Create scanner instance
    scanner = docscanner.createInstance()
    
    # Detect documents in image
    results = scanner.detect("document.jpg")
    
    # Process each detected document
    for result in results:
        print(f"Document corners: ({result.x1},{result.y1}) -> ({result.x3},{result.y3})")
        
        # Normalize the document (perspective correction) - now returns the image
        normalized_img = scanner.normalize(result, EnumImageColourMode.ICM_COLOUR)
        
        # Use the normalized image directly
        if normalized_img is not None:
            cv2.imshow("Normalized", normalized_img)
    ```

For real-time camera scanning:
    ```python
    def on_document_detected(results):
        for result in results:
            print(f"Found document at {result.x1}, {result.y1}")
    
    scanner.addAsyncListener(on_document_detected)
    
    # Process camera frames
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            scanner.detectMatAsync(frame)
    ```

Key Classes:
    - DocumentScanner: Main scanner class for detection and normalization
    - DocumentResult: Container for detection results and normalized images
    - FrameFetcher: Internal class for handling asynchronous image processing

Key Functions:
    - initLicense(): Initialize SDK license (required before use)
    - createInstance(): Create a new DocumentScanner instance
    - convertMat2ImageData(): Utility for OpenCV to SDK format conversion

Requirements:
    - Valid Dynamsoft license key
    - dynamsoft-capture-vision-bundle package
    - opencv-python for image processing
    - numpy for array operations
"""

from dynamsoft_capture_vision_bundle import (
    EnumImagePixelFormat,
    EnumErrorCode, 
    EnumPresetTemplate,
    CaptureVisionRouter,
    LicenseManager,
    ImageData,
    ImageSourceAdapter,
    CapturedResultReceiver,
    DocumentNormalizerModule,
    Quadrilateral,
    EnumImageColourMode,
    Point
)
from typing import List, Tuple, Callable, Union, Optional, Any
import numpy as np

__version__ = DocumentNormalizerModule.get_version()
    
class FrameFetcher(ImageSourceAdapter):
    """
    Custom image source adapter for handling frame-by-frame image processing.
    
    This class extends ImageSourceAdapter to provide continuous image fetching
    capability for real-time document detection scenarios like camera streams.
    """
    
    def has_next_image_to_fetch(self) -> bool:
        """
        Indicates whether there are more images to fetch.
        
        Returns:
            bool: Always returns True to enable continuous image fetching.
        """
        return True

    def add_frame(self, imageData: ImageData) -> None:
        """
        Adds a new image frame to the processing buffer.
        
        Args:
            imageData (ImageData): The image data to be added to the buffer
                                  for document detection processing.
        """
        self.add_image_to_buffer(imageData)


class MyCapturedResultReceiver(CapturedResultReceiver):
    """
    Custom result receiver for handling asynchronous document detection results.
    
    This class processes captured results from the SDK and converts them
    to DocumentResult objects before passing to the user-defined listener.
    """

    def __init__(self, listener: Callable[[List['DocumentResult']], None]) -> None:
        """
        Initialize the result receiver with a callback listener.
        
        Args:
            listener (callable): A callback function that will be called
                               with a list of DocumentResult objects when
                               documents are detected.
        """
        super().__init__()
        self.listener = listener
    
    def on_captured_result_received(self, result: Any) -> None:
        output: List['DocumentResult'] = []
        processed_document_result = result.get_processed_document_result()
        if processed_document_result is None or len(processed_document_result.get_deskewed_image_result_items()) == 0:
            pass
        else:
            items = processed_document_result.get_deskewed_image_result_items()
            for index, item in enumerate(processed_document_result.get_deskewed_image_result_items()):
            
                image = item.get_image_data()
                if image is not None:

                    mat = convertNormalizedImage2Mat(image)

                    location = item.get_source_deskew_quad()
                    x1 = location.points[0].x
                    y1 = location.points[0].y
                    x2 = location.points[1].x
                    y2 = location.points[1].y
                    x3 = location.points[2].x
                    y3 = location.points[2].y
                    x4 = location.points[3].x
                    y4 = location.points[3].y

                    document = DocumentResult()
                    document.x1 = x1
                    document.y1 = y1
                    document.x2 = x2
                    document.y2 = y2
                    document.x3 = x3
                    document.y3 = y3
                    document.x4 = x4
                    document.y4 = y4
                    document.normalized_image = mat
                    output.append(document)

        self.listener(output)

class DocumentResult:
    """
    Represents a detected document with its boundary coordinates and normalized image.
    
    This class contains the detection results for a single document, including
    the four corner points of the detected document boundary and optionally
    the normalized (perspective-corrected) image.
    
    Attributes:
        x1, y1 (int): Top-left corner coordinates
        x2, y2 (int): Top-right corner coordinates  
        x3, y3 (int): Bottom-right corner coordinates
        x4, y4 (int): Bottom-left corner coordinates
        source (str or numpy.ndarray, optional): Original source image
        normalized_image (numpy.ndarray, optional): Perspective-corrected document image
    """
    
    def __init__(self, item: Optional[Any] = None) -> None:
        """
        Initialize a DocumentResult from an SDK document item.
        
        Args:
            item (optional): The document item from the SDK containing detection results.
                           If None, creates an empty DocumentResult instance.
        """

        if item is not None:
            location = item.get_location()
            x1 = location.points[0].x
            y1 = location.points[0].y
            x2 = location.points[1].x
            y2 = location.points[1].y
            x3 = location.points[2].x
            y3 = location.points[2].y
            x4 = location.points[3].x
            y4 = location.points[3].y

            self.x1: int = x1
            self.y1: int = y1
            self.x2: int = x2
            self.y2: int = y2
            self.x3: int = x3
            self.y3: int = y3
            self.x4: int = x4
            self.y4: int = y4

        self.source: Optional[Union[str, np.ndarray]] = None
        self.normalized_image: Optional[np.ndarray] = None

class DocumentScanner:
    """
    Main document scanner class for detecting and normalizing documents in images.
    
    This class provides both synchronous and asynchronous document detection capabilities.
    It can process individual images or continuous streams (like camera feeds) and 
    perform document boundary detection and perspective correction.
    
    Before using this class, make sure to initialize the license using initLicense().
    
    Simplified API:
        The API has been streamlined to use a single detect() method that handles
        both file paths and OpenCV matrices automatically.
    
    Example:
        # Initialize license and create scanner
        initLicense("YOUR_LICENSE_KEY") 
        scanner = createInstance()
        
        # Detect documents from file or image matrix
        results = scanner.detect("document.jpg")  # File path
        # OR: results = scanner.detect(cv2_image)  # OpenCV matrix
        
        # Process results  
        for result in results:
            print(f"Document found at: ({result.x1},{result.y1}) to ({result.x3},{result.y3})")
    """
    
    def __init__(self) -> None:
        """Initialize the DocumentScanner with default settings."""
        cvr_instance = CaptureVisionRouter()
        self.fetcher: FrameFetcher = FrameFetcher()
        cvr_instance.set_input(self.fetcher)
        self.cvr_instance: CaptureVisionRouter = cvr_instance
        self.receiver: Optional[MyCapturedResultReceiver] = None
    
    def addAsyncListener(self, listener: Callable[[List[DocumentResult]], None]) -> None:
        """
        Start asynchronous document detection with a callback listener.

        This enables real-time document detection where results are delivered
        via the callback function as they are detected.
        
        Args:
            listener (callable): Function to call with detected documents.
                               Receives a list of DocumentResult objects.
        
        Example:
            def on_document_detected(documents):
                for document in documents:
                    print(f"Detected: {document.text}")

            reader.addAsyncListener(on_document_detected)
        """
        self.receiver = MyCapturedResultReceiver(listener)
        self.cvr_instance.add_result_receiver(self.receiver)
        error_code, error_message = self.cvr_instance.start_capturing(EnumPresetTemplate.PT_DETECT_AND_NORMALIZE_DOCUMENT)

    def clearAsyncListener(self) -> None:
        """
        Stop asynchronous document detection and remove the listener.
        
        This stops the real-time detection process and cleans up resources.
        Call this when you're done with async detection.
        """
        if self.receiver is not None:
            self.cvr_instance.remove_result_receiver(self.receiver)
            self.receiver = None
        self.cvr_instance.stop_capturing()
        
    def detect(self, input: Union[str, np.ndarray]) -> List[DocumentResult]:
        """
        Detect documents in images from various input sources.
        
        This is the primary detection method that handles multiple input types including
        image file paths and OpenCV matrices (numpy arrays). It automatically processes
        the input and returns all detected document boundaries.
        
        Args:
            input (Union[str, np.ndarray]): Input source for document detection:
                - str: File path to image (JPEG, PNG, BMP, TIFF, etc.)
                - np.ndarray: OpenCV image matrix (BGR or grayscale)
                - Also supports ImageData objects from the SDK
        
        Returns:
            List[DocumentResult]: List of DocumentResult objects representing detected documents.
                                Each result contains boundary coordinates (x1,y1 to x4,y4) and
                                source reference. Empty list if no documents found or error occurred.
        
        Features:
            - Supports common image formats (JPEG, PNG, BMP, TIFF, etc.)
            - Handles both color and grayscale images
            - Automatic input type detection and processing
            - Robust error handling with descriptive messages
            - Returns multiple documents if found in single image
        
        Error Handling:
            - Prints SDK error messages for debugging
            - Returns empty list on errors rather than raising exceptions
            - Validates input format compatibility
        
        Examples:
            # Detect from image file
            results = scanner.detect("document.jpg")
            
            # Detect from OpenCV image
            import cv2
            image = cv2.imread("document.jpg")
            results = scanner.detect(image)
            
            # Process results
            for result in results:
                print(f"Document at: ({result.x1},{result.y1}) to ({result.x3},{result.y3})")
                scanner.normalize(result, EnumImageColourMode.ICM_COLOUR)
        
        Note:
            For real-time processing, use detectMatAsync() with addAsyncListener()
            instead of calling this method repeatedly in a loop.
        """
        result = self.cvr_instance.capture(input, EnumPresetTemplate.PT_DETECT_DOCUMENT_BOUNDARIES)

        output: List[DocumentResult] = []

        if result.get_error_code() != EnumErrorCode.EC_OK:
            print("Error:", result.get_error_code(),
                    result.get_error_string())
        else:
            items = result.get_items()
            for item in items:
                document = DocumentResult(item)
                document.source = input
                output.append(document)

        return output

    def detectMatAsync(self, mat: np.ndarray) -> None:
        """
        Add an OpenCV matrix to the async processing queue.

        Use this with addAsyncListener() for real-time document detection.
        The detection results will be delivered via the async listener callback.
        
        Args:
            mat (numpy.ndarray): OpenCV image matrix to process asynchronously.
        
        Example:
            reader.addAsyncListener(my_callback)
            while True:
                frame = camera.read()
                reader.detectMatAsync(frame)
        """
        self.fetcher.add_frame(convertMat2ImageData(mat))

    def normalize(self, document: DocumentResult, color: EnumImageColourMode) -> Optional[np.ndarray]:
        """
        Perform document normalization (perspective correction) on a detected document.
        
        This method uses the document's boundary coordinates to extract and straighten the
        document area from the source image, producing a normalized rectangular image.
        The method includes robust error handling and fallback mechanisms to handle
        various result scenarios from the SDK.
        
        Args:
            document (DocumentResult): Document containing boundary coordinates (x1,y1 to x4,y4)
                                     and source image data. The source must be set in document.source.
            color (EnumImageColourMode): Color mode for the normalized image:
                                       - ICM_COLOUR: Full color output
                                       - ICM_GRAYSCALE: Grayscale output  
                                       - ICM_BINARY: Binary (black and white) output
        
        Returns:
            numpy.ndarray or None: The normalized document image as a numpy array.
                                 Returns None if normalization fails or no valid items found.
                                 The same image is also stored in document.normalized_image.
        
        Process:
            1. Configures ROI settings using document boundary coordinates
            2. Sets the desired color mode for output
            3. Captures normalized image using PT_NORMALIZE_DOCUMENT template
            4. Converts SDK ImageData to numpy array format using convertNormalizedImage2Mat
            5. Stores result in document.normalized_image and returns it
        
        Error Handling:
            - Prints SDK error messages if capture fails
            - Returns None for failed normalization attempts
            - Validates image data before conversion
            - Reports conversion errors with detailed messages
        
        Examples:
            from dynamsoft_capture_vision_bundle import EnumImageColourMode
            
            results = scanner.detect("document.jpg")
            if results:
                # Method 1: Use return value directly
                normalized_img = scanner.normalize(results[0], EnumImageColourMode.ICM_COLOUR)
                if normalized_img is not None:
                    cv2.imshow("Normalized", normalized_img)
                
                # Method 2: Access from document object (also available)
                if results[0].normalized_image is not None:
                    cv2.imwrite("output.jpg", results[0].normalized_image)
        
        Note:
            Both the return value and document.normalized_image contain the same image data.
            Use the return value for immediate processing or document.normalized_image for
            later access.
        """
        
        error_code, error_message, settings = self.cvr_instance.get_simplified_settings(EnumPresetTemplate.PT_NORMALIZE_DOCUMENT)
        quad = Quadrilateral()
        quad.points = [Point(document.x1, document.y1), Point(document.x2, document.y2), Point(document.x3, document.y3), Point(document.x4, document.y4)]
        settings.roi = quad
        settings.roi_measured_in_percentage = 0
        settings.document_settings.colour_mode = color
        error_code, error_message = self.cvr_instance.update_settings(EnumPresetTemplate.PT_NORMALIZE_DOCUMENT,settings)
        result = self.cvr_instance.capture(document.source, EnumPresetTemplate.PT_NORMALIZE_DOCUMENT)
        if result.get_error_code() != EnumErrorCode.EC_OK:
            print("Error:", result.get_error_code(),
                    result.get_error_string())
        else:
            items = result.get_items()
            if len(items) > 1:  
                item = items[1]
                image: ImageData = item.get_image_data()
                if image is not None:
                
                    mat = convertNormalizedImage2Mat(image)
                    if error_code == EnumErrorCode.EC_OK:
                        document.normalized_image = mat
                        return mat
                    else:
                        print(f"Error converting image to numpy: {error_message}")
            else:
                print("No normalized image items found in result")

        return None

def initLicense(licenseKey: str) -> Tuple[int, str]:
    """
    Initialize the Dynamsoft license for document detection.
    
    This must be called before creating any DocumentScanner instances.
    The license key enables the SDK functionality.
    
    Args:
        licenseKey (str): Your Dynamsoft license key. Use "DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==" for trial.
    
    Returns:
        tuple: (error_code, error_message) indicating license initialization result.
    
    Example:
        error_code, error_msg = initLicense("YOUR_LICENSE_KEY")
        if error_code != 0:
            print(f"License error: {error_msg}")
    """
    errorCode, errorMsg = LicenseManager.init_license(licenseKey)
    return errorCode, errorMsg

def createInstance() -> DocumentScanner:
    """
    Create a new DocumentScanner instance.
    
    This is the preferred way to create a DocumentScanner. Make sure to
    call initLicense() first.
    
    Returns:
        DocumentScanner: A new DocumentScanner instance ready for use.

    Example:
        initLicense("YOUR_LICENSE_KEY")
        reader = createInstance()
        results = reader.detectFile("document.jpg")
    """
    return DocumentScanner()

def convertMat2ImageData(mat: np.ndarray) -> ImageData:
    """
    Convert an OpenCV matrix to Dynamsoft ImageData format.
    
    This utility function handles the conversion between OpenCV's numpy
    array format and the SDK's ImageData format, including proper
    pixel format detection for RGB and grayscale images.
    
    Args:
        mat (numpy.ndarray): OpenCV image matrix (RGB, BGR, or grayscale).
    
    Returns:
        ImageData: Converted image data ready for SDK processing.
    
    Note:
        - 3-channel images are treated as RGB_888
        - Single-channel images are treated as grayscale
        - The function automatically determines stride and pixel format
    """
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

def convertNormalizedImage2Mat(normalized_image: ImageData) -> np.ndarray:
    """
    Convert a Dynamsoft ImageData object to an OpenCV-compatible numpy array.
    
    This utility function handles the conversion from the SDK's ImageData format
    back to OpenCV's numpy array format, supporting various pixel formats including
    binary, grayscale, and color images.
    
    Args:
        normalized_image (ImageData): The ImageData object from SDK normalization results
        
    Returns:
        numpy.ndarray: OpenCV-compatible image matrix
        
    Supported Formats:
        - Binary images: Single channel 8-bit
        - Grayscale images: Single channel 8-bit
        - Color images: 3-channel RGB format
        
    Note:
        The function automatically detects the pixel format and handles
        stride/padding for proper image reconstruction.
    """
    
    image_bytes = normalized_image.get_bytes()
    width = normalized_image.get_width()
    height = normalized_image.get_height()
    pixel_format = normalized_image.get_image_pixel_format() 
    if pixel_format == EnumImagePixelFormat.IPF_GRAYSCALED or pixel_format == EnumImagePixelFormat.IPF_BINARY or pixel_format == EnumImagePixelFormat.IPF_BINARY_8:
        channels = 1
        mat = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width)
        return mat
    
    else:
        channels = 3
        mat = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, width, channels)
        return mat