import sys
import cv2
import numpy as np
import hw1_3 as AP
import math

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board2.jpg'
src=cv2.imread(filename)

crop=AP.automatic_perspective(src.copy()) #color
mid=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
_,thr=cv2.threshold(mid,150,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)          
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thr)
dst=crop.copy()
idx=np.argsort(stats[:,4])[::-1]
stats=stats[idx]
print(stats)
for i in range(1,cnt):
    (x,y,w,h,area)=stats[i]
    # if area<20:
    #     continue
    
    pt1=(x,y)
    pt2=(x+w,y+h)
    cv2.rectangle(dst,pt1,pt2,(0,255,255))



cv2.imshow('src',crop)
cv2.imshow('thre',thr)
cv2.imshow('thr',dst)
cv2.waitKey()
cv2.destroyAllWindows