import cv2
import numpy as np
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MIX = 0
    
    def get(self):
        if Color(self.value) == Color.RED:
            return np.array([[[0,0,255]]])
        elif Color(self.value) == Color.GREEN:
            return np.array([[[0,255,0]]])
        elif Color(self.value) == Color.YELLOW:
            return np.array([[[0,255,255]]])
        elif Color(self.value) == Color.BLUE:
            return np.array([[[255,0,0]]])
        else:
            return None
    

def cvtLab(img):
    img = np.float32(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2Lab))
    img[:,:,0]  *=100/255
    img[:,:,1]  -= 128
    img[:,:,2]  -= 128
    return img
