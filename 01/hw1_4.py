import sys
import cv2
import numpy as np
import hw1_3 as APT

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board1.jpg'
src=cv2.imread(filename)

crop=APT.automatic_perspective(src.copy())
crop=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

_,thr=cv2.threshold(crop,100,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)


alpha=0.1
abs = cv2.convertScaleAbs(crop,alpha=1+alpha,beta=-128*alpha)

circles=cv2.HoughCircles(abs, cv2.HOUGH_GRADIENT,1,50,param1=100,param2=20,minRadius=20,maxRadius=30)
dst=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
center=[]
if circles is not None:
    for i in range(circles.shape[1]):
        cx,cy,radius=circles[0][i]
        cv2.circle(dst,(int(cx),int(cy)),int(radius),(0,0,255),2,cv2.LINE_AA)
        center.append((int(cx),int(cy),radius))

white=0
black=0
for i in center:
    pixel=thr[i[1],i[0]]
    if pixel==255:
        white+=1
    else:
        black+=1
print('w:{white}, b:{black}'.format(white=white, black=black))

cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()