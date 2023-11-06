import cv2
import numpy as np

# 이미지 경로
image_path = 'path_to_your_checkerboard_image.jpg'

# 이미지를 불러오고 그레이스케일로 변환
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 체커보드의 각 칸을 찾기 위한 모서리 검출
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

# 검출된 각 모서리에 대해 작은 사각형을 그려서 시각화 (디버깅 용도)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

# 각 모서리가 속한 칸을 찾아 말의 존재 여부와 색깔을 판별
# 여기서는 각 칸의 중심을 기준으로 색깔을 판별하는 간단한 방법을 사용합니다.

# 말의 수를 저장할 변수 초기화
white_pieces = 0
black_pieces = 0

# 체커보드의 각 칸을 순회하면서 말의 존재 여부와 색깔을 검사
for row in range(8):  # 8x8 체커보드를 가정
    for col in range(8):
        # 칸의 중심 좌표를 계산 (이 부분은 체커보드의 실제 크기와 위치에 맞추어 조정해야 합니다)
        center_x = int(x_start + col * square_width + square_width / 2)
        center_y = int(y_start + row * square_height + square_height / 2)
        
        # 칸의 중심에서의 색깔을 검사
        color = image[center_y, center_x]
        
        # 밝은 색과 어두운 색 말을 구분하는 기준 설정
        if np.mean(color) > brightness_threshold:
            white_pieces += 1
        else:
            black_pieces += 1

# 결과 출력
print(f'w:{white_pieces} b:{black_pieces}')

# 이미지 출력 (디버깅 용도)
cv2.imshow('Checkerboard Pieces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
