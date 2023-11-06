import sys
import cv2
import numpy as np


if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board2.jpg'

src=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed!')
    exit()
src = cv2.resize(src,(640,640),interpolation=cv2.INTER_LANCZOS4)
alpha=0.1
abs = cv2.convertScaleAbs(src,alpha=1+alpha,beta=-128*alpha)

circles=cv2.HoughCircles(abs, cv2.HOUGH_GRADIENT,1,50,param1=100,param2=20,minRadius=20,maxRadius=35)
dst=cv2.cvtColor(src,cv2.COLOR_GRAY2BGR)

if circles is not None:
    for i in range(circles.shape[1]):
        cx,cy,radius=circles[0][i]
        cv2.circle(dst,(int(cx),int(cy)),int(radius),(0,0,255),2,cv2.LINE_AA)

cv2.imshow('src',src)
cv2.imshow('abs',abs)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()