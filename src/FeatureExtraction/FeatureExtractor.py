'''
Created on 06-Aug-2017

@author: sreram
'''

import numpy as np
import cv2
import copy
import ImageProcessExtension

class FeatureExtract:
    
    
    GREEN_HSV = (cv2.cvtColor(np.uint8([[[0,255,0 ]]]), cv2.COLOR_BGR2HSV)) [0][0]
    
    BLUE_HSV  = (cv2.cvtColor(np.uint8([[[255,0,0 ]]]), cv2.COLOR_BGR2HSV)) [0][0]
    
    GREEN_LOW  = GREEN_HSV[0] - 10;
    GREEN_HIGH = GREEN_HSV[0] + 10;
    
    BLUE_LOW   = BLUE_HSV[0]  - 10;
    BLUE_HIGH  = BLUE_HSV[0]  + 10;
    
    
    RED_EXTRACT    =  [ 
                        [
                            np.array([  0,  50, 50]), 
                            np.array([ 10, 255,255])
                        ], 
                        
                        [
                            np.array([170, 50, 50]), 
                            np.array([180,255,255])
                        ]
                      ]
    
    GREEN_EXTRACT  = [
                        np.array([ GREEN_LOW,   50,  50]), 
                        np.array([ GREEN_HIGH, 255, 255])
                     ]
    
    BLUE_EXTRACT   = [
                        np.array([ BLUE_LOW,   50,  50]), 
                        np.array([ BLUE_HIGH, 255, 255])
                     ]
    
    AGGREGATE_TYPE_MAJOR = 255
    AGGREGATE_TYPE_MINOR = 0
    
    def __init__ (self, frameListOrder, exaggerateOrder = 2):
        
        
        self.__extColByOrder = ImageProcessExtension.ImageProcessExtension()
        
        self.__curImage = None
            
        self.__dispFrameList = []  
        self.__frameListOrder = frameListOrder
        self.__exaggerateOrder = exaggerateOrder
        
        
    def setFrameListOrder (self, order):
        self.__frameListOrder = order
        
    def setExaggerateOrder(self, order):
        self.__exaggerateOrder = order
        
    def getCurImage (self):
        return self.__curImage
    
    def exaggerateColorByOrder (self, order):
        frame_list = self.__extColByOrder.exaggerateColorByOrder(self.__curImage, order)
        self.__curImage = np.asarray(np.uint8(frame_list))
        
    
    def extractColor (self, colorRange):
        
        if isinstance(colorRange[0], list) is False:
            self.__curImage = cv2.inRange(self.__curImage, colorRange[0], colorRange[1])
        else:
            redRegionOfImage0 = cv2.inRange(self.__curImage, colorRange[0][0], colorRange[0][1])
            redRegionOfImage1 = cv2.inRange(self.__curImage, colorRange[1][0], colorRange[1][1])
            redRegionOfImage  = redRegionOfImage0 + redRegionOfImage1
            self.__curImage   = redRegionOfImage
        
    
    def aggregateImage (self, aggType = AGGREGATE_TYPE_MAJOR):
        for i in range(1, self.__frameListOrder):
            self.__dispFrameList[0][np.where(self.__dispFrameList[i] == aggType)] = aggType
        self.__curImage = self.__dispFrameList[0]
        self.__dispFrameList = []
    
        
    def dispFrameListSize(self):
        return len(self.__dispFrameList)
    
    def ___extractFeatureDefault (self, 
                               videoCaptureInstance, 
                               colorRange, 
                               aggregateType):
                
        for i in range(0, self.__frameListOrder):
            
            ret, self.__curImage = videoCaptureInstance.read()
            self.__curImage = cv2.GaussianBlur(self.__curImage, (15,15), 0)
            
            if self.__exaggerateOrder >= 2:
                self.exaggerateColorByOrder(self.__exaggerateOrder)
            
            self.__curImage = cv2.cvtColor(self.__curImage, cv2.COLOR_BGR2HSV)
            self.extractColor(colorRange)        
            self.__dispFrameList.append(self.__curImage)

        self.aggregateImage(aggregateType)
            
    
    def configureDefaultFeatureExtractor (self):
        self.extractFeature = self.___extractFeatureDefault
        
    def copy (self):
        return copy.deepcopy(self)
    
    
    def getHeightWidth (self):
        return (self.__height, self.__width)
    
    pass

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    fExt = FeatureExtract(3, 0)     
    fExt.configureDefaultFeatureExtractor()
    
    
    #EXPERIMENT:
    
    def updateFrame (rectDrawStack, img, frameName):
        draw_rectangle_stack(rectDrawStack, img )      
        cv2.imshow(frameName, img)
    
    def draw_rect(event,x,y,flags,param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            draw_rect.pos1 = (x,y)
            
        if event == cv2.EVENT_LBUTTONUP:    
            
            draw_rect.drawStack.append((draw_rect.pos1, 
                                          (x,y), 
                                          draw_rect.color, 
                                          draw_rect.thickness ))
            draw_rect.pos1 = None
            
            
            
    draw_rect.drawStack = []
    draw_rect.color     = (170,170,170)
    draw_rect.thickness = 2
    draw_rect.image     = None
    draw_rect.frameName = 'frame'
    draw_rect.pos1      = None

    
    
    def draw_rectangle_stack (drawStack, img):
        for i in range(len(drawStack)):
            cv2.rectangle (img,              # image
                           drawStack[i][0],  # position 1
                           drawStack[i][1],  # position 2
                           drawStack[i][2],  # color
                           drawStack[i][3]   # thickness
                           )
            
    
    
    #-------------------------------------------------------------
    
    
    while (True):
        
        fExt.extractFeature(cap, 
                            FeatureExtract.RED_EXTRACT, 
                            FeatureExtract.AGGREGATE_TYPE_MAJOR)
        
        draw_rect.image = fExt.getCurImage().copy()
        
        cv2.setMouseCallback(draw_rect.frameName, draw_rect)
          
        #draw_rectangle_stack(draw_rect.drawStack, draw_rect.image )      
        #cv2.imshow('frame', draw_rect.image)
        
        updateFrame (draw_rect.drawStack, draw_rect.image, 'frame')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
            
            