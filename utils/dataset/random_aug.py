import random
import numpy as np
import cv2 as cv


class huohua:
    def __init__(self,loc:tuple):
        
        self.center = loc
        self.num = random.randint(10,200)
        self.range = random.uniform(np.pi/3,np.pi)
        self.angle = random.uniform(0, np.pi)
        self.huoxins = []
        self.generatehuoxin()

    def generatehuoxin(self):
        for i in range(0,self.num):
            angle = random.uniform(0,self.range)+self.angle
            k1 = random.uniform(0,2)
            k2 = random.uniform(k1,2)
            w = random.uniform(1,20)
            self.huoxins.append([angle,k1,k2,w])

class argHuoHua:
    def __init__(self,img_shape:tuple):
        
        self.center = (np.random.uniform(0,img_shape[0]),np.random.uniform(0,img_shape[1]))
        self.imw,self.imh,_ = img_shape
        self.num = random.randint(10,200)
        self.range = random.uniform(np.pi/3,np.pi)
        self.angle = random.uniform(0, np.pi)
        self.huoxins = []
        self.generatehuoxin()

    def generatehuoxin(self):
        for i in range(0,self.num):
            angle = random.uniform(0,self.range)+self.angle
            k1 = random.uniform(0,2)
            k2 = random.uniform(k1,2)
            w = random.uniform(1,20)
            self.huoxins.append([angle,k1,k2,w])

    def argImg(self,im):
        for h in self.huoxins:
            p1 = (int(self.center[0]+h[1]*np.cos(h[0])*self.imw),int(self.center[1]+h[1]*np.sin(h[0])*self.imh))
            p2 = (int(self.center[0]+h[2]*np.cos(h[0])*self.imw),int(self.center[1]+h[2]*np.sin(h[0])*self.imh))
            cv.line(im,p1,p2,255,int(h[3]))
        return im

