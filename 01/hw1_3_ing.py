import sys
import cv2
import numpy as np
import math

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board2.jpg'

def auto_canny(image, sigma=0.33):
    # Canny 엣지 검출기를 위한 자동 임계값 계산
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)
   
src = cv2.imread(filename)
dst=src.copy()
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
edged = auto_canny(blurred)
cv2.imshow('edge',edged)

cordinate=[]
lines=cv2.HoughLinesP(edged,1,math.pi/180,150,minLineLength=100, maxLineGap=50)
if lines is not None:
    for i in range(lines.shape[0]):
        pt1=(lines[i][0][0],lines[i][0][1])
        pt2=(lines[i][0][2],lines[i][0][3])
        cordinate.append((pt1,pt2))
        cv2.line(dst,pt1,pt2,(0,0,255),2,cv2.LINE_AA)
# lines = cv2.HoughLines(edged, 1, np.pi / 180, 150)
# print(len(lines))
# if lines is not None:
#     cordinate=[]
#     for i in range(lines.shape[0]):
#         rho=lines[i][0][0]
#         theta=lines[i][0][1]
#         cos_t=math.cos(theta)
#         sin_t=math.sin(theta)
#         x0, y0 = rho*cos_t, rho*sin_t
#         alpha=1000
#         pt1=(int(x0-alpha*sin_t),int(y0+alpha*cos_t))
#         pt2=(int(x0+alpha*sin_t),int(y0-alpha*cos_t))
#         cordinate.append((pt1,pt2))
#         cv2.line(dst, pt1,pt2, (0, 0, 255), 2,cv2.LINE_AA)
print(len(lines))
print(lines)
print(cordinate)
print(cordinate[0][0][0])
cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('ed',edged)
cv2.waitKey()
cv2.destroyAllWindows