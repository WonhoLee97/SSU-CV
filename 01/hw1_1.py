import sys
import cv2
import numpy as np
import hw1_3 as APT
import math

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board1.jpg'
src=cv2.imread(filename)

crop=APT.automatic_perspective(src.copy()) #color
mid=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(mid, 50,150)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 230)

if lines is not None:
    cord_x=[]
    cord_y=[]
    for i in range(lines.shape[0]):
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        cos_t=math.cos(theta)
        sin_t=math.sin(theta)
        x0, y0 = rho*cos_t, rho*sin_t
        alpha=1000
        pt1=(int(x0-alpha*sin_t),int(y0+alpha*cos_t))
        pt2=(int(x0+alpha*sin_t),int(y0-alpha*cos_t))
        cord_x.append(pt1[0])
        cord_x.append(pt2[0])
        cord_y.append(pt1[1])
        cord_y.append(pt2[1])
        cv2.line(crop, pt1,pt2, (0, 0, 255), 2,cv2.LINE_AA)

for i in range(len(cord_x)):
    if cord_x[i]>640:
        cord_x[i]=640
    elif cord_x[i]<0:
        cord_x[i]=0
for i in range(len(cord_y)):
    if cord_y[i]>640:
        cord_y[i]=640
    elif cord_y[i]<0:
        cord_y[i]=0

top=0
left=0
right=0
bottom=0

for i in range(len(cord_x)): #전체좌표
    if cord_y[i]<5: #상단
        top+=1
        cv2.circle(crop,(cord_x[i],cord_y[i]),3,(255,0,255),5)
    if cord_y[i]>635: #하단
        bottom+=1
        cv2.circle(crop,(cord_x[i],cord_y[i]),3,(255,255,255),5)
    if cord_x[i]<5: #좌측
        left+=1
        cv2.circle(crop,(cord_x[i],cord_y[i]),3,(0,255,255),5)
    if cord_x[i]>635: #우측
        right+=1
        cv2.circle(crop,(cord_x[i],cord_y[i]),3,(255,255,0),5) 
value=[top,left,right,bottom]
result=sum(value)/4
if result>15: #오차를 감안한 Threshold
    print('10x10')
else:
    print('8x8')
cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows