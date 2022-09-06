from time import sleep
import docscanner
import numpy as np
import cv2
import time

# set license
docscanner.initLicense("DLS2eyJoYW5kc2hha2VDb2RlIjoiMjAwMDAxLTE2NDk4Mjk3OTI2MzUiLCJvcmdhbml6YXRpb25JRCI6IjIwMDAwMSIsInNlc3Npb25QYXNzd29yZCI6IndTcGR6Vm05WDJrcEQ5YUoifQ==")

scanner = docscanner.createInstance()

ret = scanner.setParameters(docscanner.Templates.color)
print(ret)

def showNormalizedImage(name, normalized_image):
    mat = docscanner.convertNormalizedImage2Mat(normalized_image)
    cv2.imshow(name, mat)
    return mat
    
# detectFile()
print('')
print('Test detectFile()')
results = scanner.detectFile("images/1.png")
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
    
    normalized_image = scanner.normalizeFile("images/1.png", x1, y1, x2, y2, x3, y3, x4, y4)
    normalized_image.recycle()
    # showNormalizedImage("Normalized Image", normalized_image)
    # cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
# cv2.imshow('Document Image', image)
# cv2.waitKey(0)


# detectMat()
# print('')
# print('Test detectMat()')

# image = cv2.imread("images/1.png")
# results = scanner.detectMat(image)
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
#     normalized_image.recycle()
#     cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
# cv2.imshow('Document Image', image)
# cv2.waitKey(0)

# detectMatAsync()
# print('')
# print('Test detectMatAsync()')
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
    
#     cv2.imshow('Document Image', image)  
#     cv2.waitKey(0)
    
# import cv2
# image = cv2.imread("images/1.png")
# scanner.addAsyncListener(callback)
# scanner.detectMatAsync(image)
# sleep(5)

# g_results = None
# g_normalized_images = []

# def callback(results):
#     global g_results
#     g_results = results

# scanner.addAsyncListener(callback)

# cap = cv2.VideoCapture(0)
# while True:
#     ret, image = cap.read()
    
#     ch = cv2.waitKey(1)
#     if ch == 27:
#         break
#     elif ch == ord('n'): # normalize image
#         if g_results != None:
#             g_normalized_images = []
#             index = 0
#             for result in g_results:
#                 x1 = result.x1
#                 y1 = result.y1
#                 x2 = result.x2
#                 y2 = result.y2
#                 x3 = result.x3
#                 y3 = result.y3
#                 x4 = result.x4
#                 y4 = result.y4
                
#                 normalized_image = scanner.normalizeBuffer(image, x1, y1, x2, y2, x3, y3, x4, y4)
#                 g_normalized_images.append((str(index), normalized_image))
#                 mat = showNormalizedImage(str(index), normalized_image)
#                 index += 1
#     elif ch == ord('s'): # save image
#         for data in g_normalized_images:
#             # cv2.imwrite('images/' + str(time.time()) + '.png', image)
#             cv2.destroyWindow(data[0])
#             data[1].save(str(time.time()) + '.png')
#             print('Image saved')
#             data[1].recycle()
            
#         g_normalized_images = []
        
#     if image is not None:
#         scanner.detectMatAsync(image)
    
#     if g_results != None:
#         for result in g_results:
#             x1 = result.x1
#             y1 = result.y1
#             x2 = result.x2
#             y2 = result.y2
#             x3 = result.x3
#             y3 = result.y3
#             x4 = result.x4
#             y4 = result.y4
            
#             cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
        
#     cv2.putText(image, 'Press "n" to normalize image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#     cv2.putText(image, 'Press "s" to save image', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#     cv2.putText(image, 'Press "ESC" to exit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#     cv2.imshow('Document Scanner', image)