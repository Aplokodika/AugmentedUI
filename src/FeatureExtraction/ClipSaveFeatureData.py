'''
Created on 11-Aug-2017

@author: sreram
'''

import numpy as np
import cv2
import copy
import ImageProcessExtension
from FeatureExtractor import FeatureExtract

class ClipFeature:
    
    def __init__ (self, featureExtractInstance):
        self.__fExt = featureExtractInstance
        
    def writeClipToCSVFile (self, pos1, pos2, img, fileName):
        cropped_image = img [pos1[0] : pos1[1], pos2[0] : pos2[1]];
        
        pass
        


if __name__ == "__main__":
    cFeature = ClipFeature(2,0)