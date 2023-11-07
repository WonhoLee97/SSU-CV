import sys
import cv2
import numpy as np
import hw1_3 as AP
import math

def auto_canny(image, sigma=0.33):
    # Canny 엣지 검출기를 위한 자동 임계값 계산
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)
def line_similarity(line1, line2, angle_threshold, distance_threshold):
    # 두 직선의 기울기와 절편 계산
    for (x1, y1, x2, y2) in [line1, line2]:
        if x2 - x1 == 0:  # 분모가 0이 되는 경우를 방지
            m = np.inf
            c = np.inf
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
        if 'm1' in locals():
            m2, c2 = m, c
        else:
            m1, c1 = m, c

    # 두 직선 간의 각도 계산
    if m1 != np.inf and m2 != np.inf:
        tan_theta = abs((m2 - m1) / (1 + m1 * m2))
        angle_diff = np.arctan(tan_theta) * (180 / np.pi)
    else:
        angle_diff = 0

    # 두 직선 간의 최소 거리 계산
    if c1 != np.inf and c2 != np.inf:
        distance = abs(c2 - c1) / np.sqrt(1 + m1**2)
    else:
        distance = 0

    # 각도와 거리 기반 유사성 결정
    if angle_diff < angle_threshold and distance < distance_threshold:
        return True
    else:
        return False

def filter_similar_lines(lines, angle_threshold=10, distance_threshold=10):
    filtered_lines = []
    for line in lines:
        if not any(line_similarity(line, other_line, angle_threshold, distance_threshold) for other_line in filtered_lines):
            filtered_lines.append(line)
    return filtered_lines
if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board2.jpg'
src=cv2.imread(filename)

crop=AP.automatic_perspective(src.copy())
# crop=src.copy()
dst=crop.copy()

crop=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(crop,(5,5),0)
edged = auto_canny(blurred)
cv2.imshow('edge',edged)
mor=cv2.morphologyEx(edged,cv2.MORPH_GRADIENT,None,anchor=(-1,-1),iterations=2)
cordinate=[]
lines=cv2.HoughLinesP(mor,1,math.pi/180,110,minLineLength=300, maxLineGap=30)
if lines is not None:
    unique_lines = filter_similar_lines(lines[:, 0], angle_threshold=10, distance_threshold=10)
    for x1, y1, x2, y2 in unique_lines:
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 5,cv2.LINE_AA)
    # for i in range(lines.shape[0]):
    #     pt1=(lines[i][0][0],lines[i][0][1])
    #     pt2=(lines[i][0][2],lines[i][0][3])
    #     cordinate.append((pt1,pt2))
    #     cv2.line(dst,pt1,pt2,(255,0,0),1,cv2.LINE_AA)
print(lines)
print(len(lines))
print(len(unique_lines))
cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('mor',mor)
cv2.waitKey()
cv2.destroyAllWindows