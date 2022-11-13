import numpy as np
import time
import cv2
from dotenv import load_dotenv
import os

class ImageProcessor:

    def __init__(self, target_width, target_height, technique, grayscale = False):
        self.target_width = target_width
        self.target_height = target_height
        self.grayscale = grayscale
        self.technique = technique

    def change_dimension(self, img):
        if self.technique == "SIMPLE":
            img = cv2.resize(img, (self.target_width, self.target_height), interpolation = cv2.INTER_AREA)
        return img
    
    def process(self, img_path):
        if self.grayscale:
            img = cv2.imread(img_path, 0) # 0 flag for reading image in gray scale
        else:
            img = cv2.imread(img_path) # reading image in all 3 channels
        img = self.change_dimension(img)
        return img

    def get_height(self, img):
        return img.shape[1]

    def get_width(self, img):
        return img.shape[0]



if __name__ == "__main__":
    start_time = time.time() # start time
    load_dotenv() # load environment variables from .env file
    print("Fetching dataset from "+os.getenv("DATASET_IMAGES_PATH"))
    os.chdir(os.getenv("DATASET_IMAGES_PATH")) # change dir to that of dataset
    files = os.listdir() # list all the files and folders
    count = 0
    img_processor = ImageProcessor(int(os.getenv("TARGET_WIDTH")), int(os.getenv("TARGET_HEIGHT")), os.getenv("RESIZING_TECHNIQUE"))

    # create output directory if it does not exists
    output_path = os.getenv("PROCESSED_DATASET_PATH")+os.getenv("RESIZING_TECHNIQUE")+"\\"
    print(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok = True)
    
    for f in files:
        if f[-3:]==os.getenv("IMAGE_FORMAT"):
            output = img_processor.process(os.getenv("DATASET_IMAGES_PATH")+f)
            cv2.waitKey(0)
            cv2.imwrite(output_path+f, output)
            
    end_time = time.time() # end time
    print("Execution time: "+str(end_time-start_time)+" seconds")


