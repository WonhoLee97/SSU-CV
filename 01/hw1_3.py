import sys
import cv2
import numpy as np

def order_points(pts):
    src_pts=np.zeros((4,2),dtype="float32")
    s=pts.sum(axis=1)
    src_pts[0]=pts[np.argmin(s)]
    src_pts[2]=pts[np.argmax(s)]
    
    diff=np.diff(pts,axis=1)
    src_pts[1]=pts[np.argmin(diff)]
    src_pts[3]=pts[np.argmax(diff)]

    return src_pts

def get_pers_mat(src, pts):
    src_pts=order_points(pts)
    (tl,tr,br,bl)=src_pts
    w=640
    h=640
    dst_pts=np.array([[0,0],
                      [w-1,0],
                      [w-1,h-1],
                      [0,h-1]]).astype(np.float32)
    pers_mat=cv2.getPerspectiveTransform(src_pts,dst_pts)
    return pers_mat

def automatic_perspective(src):
    gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,50,150)
    contours,_=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)[:5] #크기가 큰 순서대로 정렬
    for pts in contours:
        approx=cv2.approxPolyDP(pts,cv2.arcLength(pts,True)*0.02,True)
        tmp=approx
        if len(approx)==4: #가장 큰 사각형을 보드라고 가정
            tmp=approx
            break
    pers_mat=get_pers_mat(src,tmp.reshape(4,2))
    dst=cv2.warpPerspective(src,pers_mat,(640,640))
    return dst

if __name__=='__main__':
    if len(sys.argv)>1:
        filename=sys.argv[1]
    else:
        filename='board1.jpg'
    src=cv2.imread(filename)
    dst=automatic_perspective(src)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
