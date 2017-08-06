'''
Created on 06-Aug-2017

@author: sreram
'''

import numpy as np
import cv2
import copy


class FeatureExtract:
    
    
    GREEN_HSV = (cv2.cvtColor(np.uint8([[[0,255,0 ]]]), cv2.COLOR_BGR2HSV)) [0][0]
    
    BLUE_HSV  = (cv2.cvtColor(np.uint8([[[255,0,0 ]]]), cv2.COLOR_BGR2HSV)) [0][0]
    
    RED_EXTRACT =  [ [[0,  50, 50], [10, 255,255]], 
                     [[170,50, 50], [180,255,255]]]
    
    GREEN_EXTRACT = [[ (FeatureExtract.GREEN_HSV[0] - 10), 50,  50 ], 
                     [ (FeatureExtract.GREEN_HSV[0] + 10), 255, 255]]
    
    BLUE_EXTRACT  = [[ (FeatureExtract.BLUE_HSV[0]  - 10), 50,  50], 
                     [ (FeatureExtract.BLUE_HSV[0]  + 10), 255, 255]]
    
    def __init__ (self, height, width, curImage, imageModified = None):
        
        self.__height = height
        self.__width  = width
        
        self.__curImage = curImage
        
        if imageModified is None:
            self.__imageModified = cv2.cvtColor(self.__curImage, cv2.COLOR_BGR2HSV)
        else:
            self.__imageModified = imageModified
        pass
    
    
    def extractColor (self, extraction):
        
        if isinstance(extraction[0][0], list) :
            self.__imageModified = cv2.inRange(self.__imageModified, extraction[0], extraction[1])
        else:
            redRegionOfImage0 = cv2.inRange(self.__imageModified, extraction[0][0], extraction[0][1])
            redRegionOfImage1 = cv2.inRange(self.__imageModified, extraction[1][0], extraction[1][1])
 
            
        
        
        pass
    
    def copy (self):
        return copy.deepcopy(self)
    
    
    def getHeightWidth (self): 
        return (self.__height, self.__width)
    
    pass
