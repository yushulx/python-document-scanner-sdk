from time import sleep
import docscanner
import numpy as np
import cv2
import time

# set license
docscanner.initLicense("DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")

scanner = docscanner.createInstance()

p_binary = '''
{
    "GlobalParameter":{
        "Name":"GP"
    },
    "ImageParameterArray":[
        {
            "Name":"IP-1",
            "NormalizerParameterName":"NP-1"
        }
    ],
    "NormalizerParameterArray":[
        {
            "Name":"NP-1",
            "ColourMode": "ICM_BINARY" 
        }
    ]
}
'''

p_color = '''
{
    "GlobalParameter":{
        "Name":"GP"
    },
    "ImageParameterArray":[
        {
            "Name":"IP-1",
            "NormalizerParameterName":"NP-1"
        }
    ],
    "NormalizerParameterArray":[
        {
            "Name":"NP-1",
            "ColourMode": "ICM_COLOUR" 
        }
    ]
}
'''

p_grayscale = '''
{
    "GlobalParameter":{
        "Name":"GP"
    },
    "ImageParameterArray":[
        {
            "Name":"IP-1",
            "NormalizerParameterName":"NP-1"
        }
    ],
    "NormalizerParameterArray":[
        {
            "Name":"NP-1",
            "ColourMode": "ICM_GRAYSCALE"
        }
    ]
}
'''

ret = scanner.setParameters(p_color)
print(ret)

def showNormalizedImage(name, normalized_image):
    bytearray = normalized_image.bytearray
    width = normalized_image.width
    height = normalized_image.height
    
    channels = 3
    if normalized_image.format == docscanner.ImagePixelFormat.IPF_BINARY:
        channels = 1
        all = []
        
        for byte in bytearray:
            
            byteCount = 7
            while byteCount >= 0:
                b = (byte & (1 << byteCount)) >> byteCount
                if b == 1:
                    all.append(255)
                else:
                    all.append(0)
                    
                byteCount -= 1
            
        bytearray = all
        width = normalized_image.stride * 8
        
    elif normalized_image.format == docscanner.ImagePixelFormat.IPF_GRAYSCALED:
        channels = 1
    
    mat = np.array(bytearray, dtype=np.uint8).reshape(height, width, channels)
    cv2.imshow(name, mat)
    return mat
    
# decodeFile()
# print('')
# print('Test decodeFile()')
# results = scanner.decodeFile("images/1.png")
# image = cv2.imread("images/1.png")
# for result in results:
#     x1 = result.x1
#     y1 = result.y1
#     x2 = result.x2
#     y2 = result.y2
#     x3 = result.x3
#     y3 = result.y3
#     x4 = result.x4
#     y4 = result.y4
    
#     normalized_image = scanner.normalizeFile("images/1.png", x1, y1, x2, y2, x3, y3, x4, y4)
#     showNormalizedImage("Normalized Image", normalized_image)
#     cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
# cv2.imshow("image", image)
# cv2.waitKey(0)


# decodeMat()
# print('')
# print('Test decodeMat()')

# image = cv2.imread("images/1.png")
# results = scanner.decodeMat(image)
# for result in results:
#     x1 = result.x1
#     y1 = result.y1
#     x2 = result.x2
#     y2 = result.y2
#     x3 = result.x3
#     y3 = result.y3
#     x4 = result.x4
#     y4 = result.y4
    
#     normalized_image = scanner.normalizeBuffer(image, x1, y1, x2, y2, x3, y3, x4, y4)
#     showNormalizedImage("Normalized Image", normalized_image)
#     cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
# cv2.imshow("image", image)
# cv2.waitKey(0)

# decodeMatAsync()
# print('')
# print('Test decodeMatAsync()')
# def callback(results):
#     for result in results:
#         x1 = result.x1
#         y1 = result.y1
#         x2 = result.x2
#         y2 = result.y2
#         x3 = result.x3
#         y3 = result.y3
#         x4 = result.x4
#         y4 = result.y4
        
#         cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
#     cv2.imshow("image", image)  
#     cv2.waitKey(0)
    
# import cv2
# image = cv2.imread("images/1.png")
# scanner.addAsyncListener(callback)
# scanner.decodeMatAsync(image)
# sleep(5)

g_results = None
g_normalized_images = []

def callback(results):
    global g_results
    g_results = results

scanner.addAsyncListener(callback)

cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    if image is not None:
        scanner.decodeMatAsync(image)
    
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
            
            cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
        
    cv2.imshow('Document Scanner', image)
    ch = cv2.waitKey(1)
    if ch == 27:
        break
    elif ch == ord('n'): # normalize image
        if g_results != None:
            g_normalized_images = []
            index = 0
            for result in g_results:
                x1 = result.x1
                y1 = result.y1
                x2 = result.x2
                y2 = result.y2
                x3 = result.x3
                y3 = result.y3
                x4 = result.x4
                y4 = result.y4
                
                normalized_image = scanner.normalizeBuffer(image, x1, y1, x2, y2, x3, y3, x4, y4)
                mat = showNormalizedImage(str(index), normalized_image)
                g_normalized_images.append(mat)
                index += 1
    elif ch == ord('s'): # save image
        for image in g_normalized_images:
            cv2.imwrite('images/' + str(time.time()) + '.png', image)
            print('Image saved')