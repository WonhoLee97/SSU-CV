import cv2
import numpy as np

# 이미지를 로드합니다.
image = cv2.imread('board2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 체커보드의 교차점을 찾습니다.
# 체커보드의 모서리 수는 실제 칸의 수보다 하나 적습니다. (예: 8x8 체커보드는 7x7 교차점을 가짐)
ret, corners = cv2.findChessboardCorners(gray_image, (7,7))

# 교차점이 성공적으로 찾아졌는지 확인합니다.
# if ret:
#     # 체커보드의 사이즈를 결정합니다.
#     size = (corners.shape[0] // 7, 7)
#     print(f"The checkerboard size is: {size}")
# else:
#     print("Checkerboard was not detected.")

# 이미지에 교차점을 표시하고 결과를 보여줍니다.
cv2.drawChessboardCorners(image, (7,7), corners, ret)
cv2.imshow('Image with corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
