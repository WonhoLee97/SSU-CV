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
if len(results.pred[0])==0:
    print("False")
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("True")
    for detection in results.pred[0]:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()