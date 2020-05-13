import cv2
def img_filter(img):
    #使用中值濾波器濾波
    img_median=cv2.medianBlur(img,3)
    #使用ostu二值化影像
    _,img_median=cv2.threshold(img_median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #放大圖像
    img_median=cv2.resize(img_median,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    return img_median
#依照圖片輪廓切割圖片
def img_BoundingCut(img):
    img_arr=[]
    if img is not None:
        #canny邊緣偵測
        cnn_im=cv2.Canny(img,100,200)
        #找尋他們的contour
        contours,_=cv2.findContours(cnn_im.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sort_con=sorted([(c,cv2.boundingRect(c)[0]) for c in contours],key=lambda x:x[1])
        #篩選所要的bounding
        arr=[]
        for (c,_) in sort_con:
            (x,y,w,h)=cv2.boundingRect(c)
            if h>25 and w>25:
                arr.append((x,y,w,h))
                print(x,y)
        # 依照選定的bounding來切割圖片
        for k,(x,y,w,h) in enumerate(arr):
            roi=img[y:y+h,x:x+w]
            threshold=roi.copy()
            img_arr.append(threshold)
        return img_arr