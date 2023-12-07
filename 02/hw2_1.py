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
else:
    print("True")