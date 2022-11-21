from .docscanner import * 
import os
import json
import numpy as np

__version__ = version
    
class ImagePixelFormat:
	# 0:Black, 1:White
    IPF_BINARY = 0
    # 0:White, 1:Black
    IPF_BINARYINVERTED = 1
	# 8bit gray
    IPF_GRAYSCALED = 2
	# NV21 
    IPF_NV21 = 3
	# 16bit with RGB channel order stored in memory from high to low address
    IPF_RGB_565 = 4
	# 16bit with RGB channel order stored in memory from high to low address
    IPF_RGB_555 = 5 
    # 24bit with RGB channel order stored in memory from high to low address
    IPF_RGB_888 = 6 
    # 32bit with ARGB channel order stored in memory from high to low address
    IPF_ARGB_8888 = 7   
    # 48bit with RGB channel order stored in memory from high to low address
    IPF_RGB_161616 = 8  
    # 64bit with ARGB channel order stored in memory from high to low address
    IPF_ARGB_16161616 = 9   
    # 32bit with ABGR channel order stored in memory from high to low address
    IPF_ABGR_8888 = 10  
    # 64bit with ABGR channel order stored in memory from high to low address
    IPF_ABGR_16161616 = 11  
    # 24bit with BGR channel order stored in memory from high to low address
    IPF_BGR_888 = 12

def convertNormalizedImage2Mat(normalized_image):
    
    bytearray = normalized_image.bytearray
    width = normalized_image.width
    height = normalized_image.height
    
    channels = 3
    if normalized_image.format == ImagePixelFormat.IPF_BINARY:
        channels = 1
        all = []
        skip = normalized_image.stride * 8 - width
        
        index = 0
        n = 1
        for byte in bytearray:
            
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
        
    elif normalized_image.format == ImagePixelFormat.IPF_GRAYSCALED:
        channels = 1
    
    mat = np.array(bytearray, dtype=np.uint8).reshape(height, width, channels)
    
    return mat

class Templates:
    binary = '''
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

    color = '''
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

    grayscale = '''
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