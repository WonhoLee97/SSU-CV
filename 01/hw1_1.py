import cv2
import numpy as np

# 이미지를 불러옵니다
image = cv2.imread('board3.jpg')
if image is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러를 적용하여 이미지 노이즈 제거
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 엣지 검출
edged = cv2.Canny(blurred, 50, 150)

# 윤곽선 검출
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 윤곽선 찾기
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.1 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# 체커보드는 4개의 모서리를 가져야 하므로 4개의 점이 있는 윤곽선을 찾습니다
if len(approx) == 4:
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    cv2.imshow('Checkerboard', image)
    cv2.waitKey(0)
    
    # 모서리 포인트 추출
    pts = approx.reshape(4, 2)
    rows, cols = [], []

    # 각 모서리에서 그레이스케일 이미지의 행과 열을 따라 값의 변화를 검사하여 교차점을 찾습니다
    for pt in pts:
        row, col = pt
        row_vals = gray[row, :]
        col_vals = gray[:, col]
        rows.append(np.sum(row_vals < np.max(row_vals) / 2))
        cols.append(np.sum(col_vals < np.max(col_vals) / 2))
    
    # 각 행과 열에서 평균 교차점 수를 구합니다
    avg_rows = np.mean(rows)
    avg_cols = np.mean(cols)
    print(rows,cols)
    print(avg_rows,avg_cols)
    # 교차점 수에 따라 체커보드의 규격을 판별합니다
    if avg_rows > 9 and avg_cols > 9:
        print("국제 룰 체커보드 (10x10)")
    elif avg_rows > 7 and avg_cols > 7:
        print("영/미식 룰 체커보드 (8x8)")
    else:
        print('no')
