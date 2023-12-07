import torch
import sys
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model.conf=0.5

if len(sys.argv)>1:
    filename=sys.argv[1]
else:
    filename='empire1.jpg'
    

img = cv2.imread(filename)
results=model(img, size=640)

# 결과 추출 및 바운딩 박스 그리기
for detection in results.pred[0]:
    x1, y1, x2, y2, conf, cls = detection
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    
    # 이미지에 바운딩 박스와 라벨 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

# 결과 이미지 보여주기
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()