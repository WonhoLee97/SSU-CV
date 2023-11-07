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
src = cv2.resize(src,(640,640),interpolation=cv2.INTER_LANCZOS4)
dst=src.copy()
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
edged = auto_canny(blurred)
cv2.imshow('edge',edged)

cordinate=[]
lines=cv2.HoughLinesP(edged,1,math.pi/180,150,minLineLength=300, maxLineGap=40)
if lines is not None:
    for i in range(lines.shape[0]):
        pt1=(lines[i][0][0],lines[i][0][1])
        pt2=(lines[i][0][2],lines[i][0][3])
        cordinate.append(pt1)
        cordinate.append(pt2)
        cv2.line(dst,pt1,pt2,(0,0,255),2,cv2.LINE_AA)

print(len(lines))
print(lines)
print(cordinate)

cordinate=np.array(cordinate)
cordinate=cordinate.reshape(-1,2)
cordinate=np.float32(cordinate)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

# k-means 알고리즘을 수행하여 데이터를 4개 군집으로 분류합니다.
# labels에는 각 포인트가 속한 군집의 인덱스가 저장됩니다.
# centers에는 각 군집의 중심점이 저장됩니다.
compactness, labels, centers = cv2.kmeans(cordinate, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print(centers)
print(labels)
closest_points=[]
for i in range(4):
    # 현재 군집의 중심점을 가져옵니다.
    center = centers[i]
    print(center)
    # 현재 군집에 속한 포인트들의 인덱스를 가져옵니다.
    points_idx = np.where(labels.flatten() == i)
    print(points_idx)
    # 해당 포인트들을 가져옵니다.
    cluster_points = cordinate[points_idx]
    print(cluster_points)
    # 중심점과의 거리를 계산합니다.
    distances = np.linalg.norm(cluster_points - center, axis=1)
    print(distances)
    # 가장 가까운 거리의 인덱스를 찾습니다.
    closest_point_idx = np.argmin(distances)
    # 가장 가까운 포인트를 추가합니다.
    print(closest_point_idx)
    closest_points.append(cluster_points[closest_point_idx])

for point in closest_points:
    print(point)
    point_cordinate=tuple(map(int,point.ravel()))
    cv2.circle(dst,point_cordinate,10, (255, 0, 0), -1)
print(centers)
for center in centers:
    center_cordinate=tuple(map(int,center.ravel()))
    cv2.circle(dst, center_cordinate, 3, (0, 255, 0), -1)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('ed',edged)
cv2.waitKey()
cv2.destroyAllWindows