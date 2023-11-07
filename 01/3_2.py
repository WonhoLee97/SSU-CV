import sys
import cv2
import numpy as np

def order_points(pts):
    # 좌상단, 우상단, 우하단, 좌하단 순으로 정렬
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # 좌표 정렬
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 변환될 이미지의 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = 300

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = 300

    # 변환 후 4점의 좌표
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 투시 변환 행렬 계산
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 변환된 이미지 반환
    return warped

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='board1.jpg'
    
src = cv2.imread(filename)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('thr',thr)
edged = cv2.Canny(gray, 50, 150)
cv2.imshow('ed',edged)
# 윤곽선 찾기
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

# 윤곽선을 순회하며 체스보드 윤곽 찾기
for c in contours:
    # 윤곽선의 근사화
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 근사화된 윤곽선이 4개의 점을 가지고 있으면 (즉, 직사각형이면) 체스보드로 간주
    if len(approx) == 4:
        screenCnt = approx
        break

# 찾은 윤곽선을 기준으로 이미지를 변환
warped = four_point_transform(src, screenCnt.reshape(4, 2))

# 결과 출력
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
