import sys
import cv2
import numpy as np
import hw1_3 as AP
import math

def setLabel(img, pts, label):
    (x,y,w,h)=cv2.boundingRect(pts)
    pt1=(x,y)
    pt2=(x+w,y+h)
    cv2.rectangle(img, pt1, pt2, (0,0,255),1)
    
if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board2.jpg'
src=cv2.imread(filename)

crop=AP.automatic_perspective(src.copy())
# crop=src.copy()
dst=crop.copy()

crop=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
_, img_bin=cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
contours,_=cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for pts in contours:
    if cv2.contourArea(pts)<400:
        continue
    approx=cv2.approxPolyDP(pts,cv2.arcLength(pts,True)*0.02,True)
    vtc=len(approx)
    
    if vtc==4:
        setLabel(crop,pts,'RECT')
    else:
        lenth=cv2.arcLength(pts,True)
        area=cv2.contourArea(pts)
        ratio=4.*math.pi*area/(lenth*lenth)
        if ratio>0.85:
            setLabel(crop, pts, 'CIR')
            
    
cv2.imshow('src',crop)
cv2.waitKey()
cv2.destroyAllWindows