import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # Canny 엣지 검출기를 위한 자동 임계값 계산
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# 이미지 로드
image = cv2.imread('board2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 자동 Canny 엣지 검출기
edges = auto_canny(blurred)

# Hough 변환으로 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
print(len(lines))
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Edges', edges)
cv2.imshow('Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()