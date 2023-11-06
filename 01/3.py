import cv2
import numpy as np

def find_corners_of_largest_polygon(img):
    """ 주어진 바이너리 이미지에서 가장 큰 폴리곤의 꼭짓점을 찾습니다. """
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    return corners

# 이미지 로드
image = cv2.imread('board2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny 엣지 검출기로 엣지 찾기
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 가장 큰 폴리곤의 꼭짓점 찾기
corners = find_corners_of_largest_polygon(edges)

# Perspective transform 준비
if corners.shape[0] == 4:
    corners = corners.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")

    # 상단 왼쪽, 오른쪽, 하단 왼쪽, 오른쪽 순서로 정렬
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    # Perspective transform을 위한 행렬 얻기
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = 300

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = 300

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Perspective transform 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # 결과 출력
    cv2.imshow('Cropped Checkerboard', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("체커판의 모서리를 찾을 수 없습니다.")
