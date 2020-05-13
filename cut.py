import cv2
import os
#顯示圖形的視窗
def img_BoundingCut(file_path):
    #從指定檔案路徑讀取檔案
    for i,img_name in enumerate(os.listdir(file_path)):
        img=cv2.imread(file_path+img_name,cv2.IMREAD_GRAYSCALE)
        print(img_name)
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
            cv2.imwrite(file_path+'image{}_{}.jpg'.format(img_name,k),threshold)#存取圖片
            cv2.waitKey(0)
    cv2.destroyAllWindows()
img_BoundingCut('./model/pic/')