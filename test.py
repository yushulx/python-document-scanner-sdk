from time import sleep
import docscanner
import numpy as np
import cv2
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
            "Name":"IP-1"
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
            "Name":"IP-1"
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
            "Name":"IP-1"
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

ret = scanner.setParameters(p_binary)
print(ret)
# # decodeFile()
# print('')
# print('Test decodeFile()')
# results = scanner.decodeFile("images/3.png")
# image = cv2.imread("images/3.png")
# for result in results:
#     x1 = result.x1
#     y1 = result.y1
#     x2 = result.x2
#     y2 = result.y2
#     x3 = result.x3
#     y3 = result.y3
#     x4 = result.x4
#     y4 = result.y4
    
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
    
#     cv2.drawContours(image, [np.int0([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])], 0, (0, 255, 0), 2)
    
# cv2.imshow("image", image)
# cv2.waitKey(0)

# decodeMatAsync()
print('')
print('Test decodeMatAsync()')
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