import numpy as np
from PIL import Image
import sys
from config import GOOGLE_EMBEDDINGS_TF_HUB_URL, IMAGE_DIR_PATH, PATH_BREAK_CHAR, VECTOR_DIR_PATH

class LocalUploader:

    def __init__(self, img_to_vector_converter, tasks):
        self.img_to_vector_converter = img_to_vector_converter
        self.tasks = tasks

    def beautify_path(func):
        '''
            Decorator to beautify the path.
            - removes extra \ from the path 
        '''

        def inner(self, *args, **kwargs):
            path = args[0]
            path = path.strip(PATH_BREAK_CHAR)
            return func(self, path)

        return inner

    @beautify_path
    def extract_image_name(self, path):

        ind = path.rfind(PATH_BREAK_CHAR)
        if ind == -1:
            return path
        else:
            return path[ind+1:]

    @beautify_path
    def upload(self, path):
        print("Uploading "+path)
        raw_img = Image.open(path)
        img = np.asarray(raw_img)
        img_name = self.extract_image_name(path)
        vector = self.img_to_vector_converter.convert(raw_img)
        self.saveImage(img,IMAGE_DIR_PATH,img_name)
        self.saveVector(vector, VECTOR_DIR_PATH, img_name[0:-4])
        for task in self.tasks:
            task.execute(vector, img_name[0:-4])

    def saveImage(self, img, path, name): # saves file to local, should be overriden to upload to web
        im = Image.fromarray(np.uint8(img))
        im.save(path+name)

    def saveVector(self, vector, path, name): #saves vector to local
        np.save(path+name, vector)

