import numpy as np
import cv2 as cv
import os

# image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
def getStart(size,req):
    mid=size//2
    start=mid-req//2
    return start

def change_dimension(img_name):
    img=cv.imread(img_name)
    ht,wd,cc=img.shape
    if ht<=720 and wd<=1024:
        top=(720-ht)//2
        bottom=720-ht-top
        left=(1024-wd)//2
        right=(1024-wd-left)
        borderType=cv.BORDER_REFLECT
        img_modified = cv.copyMakeBorder( img, top, bottom, left, right, borderType)
        img_modified = cv.GaussianBlur(img_modified,(5,5),2)
        # cv.imshow('modified',img_modified)
        cv.imwrite('aaaaa.jpg',img_modified)
    elif ht>720 and wd > 1024:
        ht_start=getStart(ht,720)
        wd_start=getStart(wd,1024)
        img_modified=img[ht_start:ht_start+720,wd_start:wd_start+1024]
        # cv.imwrite('train_input/'+img_name,img_modified)
        # img_modified = cv.GaussianBlur(img_modified,(5,5),2)
        cv.imwrite('aaaab.jpg',img_modified)
    else:
        if ht>720:
            ht_start=getStart(ht,720)
            img=img[ht_start,ht_start+720]
            ht=720
            top=(720-ht)//2
            bottom=720-ht-top
            left=(1024-wd)//2
            right=(1024-wd-left)
            borderType=cv.BORDER_REFLECT
            img_modified = cv.copyMakeBorder( img, top, bottom, left, right, borderType)
            # cv.imshow('modified',img_modified)
            # cv.imwrite('train_input/'+img_name,img_modified)
            # cv.imwrite('aaaaa.jpg',img_modified)
        else:
            wd_start=getStart(wd,1024)
            img=img[:,wd_start:wd_start+1024]
            wd=1024
            top=(720-ht)//2
            bottom=720-ht-top
            left=(1024-wd)//2
            right=(1024-wd-left)
            borderType=cv.BORDER_REFLECT
            img_modified = cv.copyMakeBorder( img, top, bottom, left, right, borderType)
            # cv.imshow('modified',img_modified)
            # cv.imwrite('train_input/'+img_name,img_modified)
            # cv.imwrite('aaaaa.jpg',img_modified)
os.chdir('../test')
print(os.getcwd())
files=os.listdir()
print(len(files))
for f in files:
    if len(f)>4 and f[-4:]=='.jpg':
        change_dimension(f)
        break