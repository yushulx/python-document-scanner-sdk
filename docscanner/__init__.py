from .docscanner import * 
import os
import json
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

