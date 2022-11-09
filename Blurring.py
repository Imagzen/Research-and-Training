import cv2 as cv
import os

def blur(img_name):
    img=cv.imread(img_name)
    img_blur=cv.GaussianBlur(img,(5,5),2)
    cv.imwrite('Blurred/blur_'+img_name,img_blur)
    cv.waitKey(0)

files=os.listdir()
for f in files:
    if len(f)>4 and f[-4:]=='.jpg':
        blur(f)

