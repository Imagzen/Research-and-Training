
from search_techniques.searchers import LinearSearcher, GreedySearcher, KMeansSearching
from logger.Logger import Logger
import csv
from config import *
from converters.text_converters import GoogleTextConverter
from similarityfunctions.provider import SimilarityProvider
from similarityfunctions.similarity import CosineSimilarity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_if_exists(searcher, caption, img_name, window_size):
    '''
        Returns true if on searching 'caption', img_name is present in top window_size results
    '''
    result = searcher.search(caption)
    for i in range(min(window_size, len(result))):
        result_img_name = result[i][0]
        if(result_img_name == img_name):
            return True
        
    return False

def test(searcher, img_names, img_captions, window_size):
    count = 0
    size = len(img_names)
    for i in range(len(img_names)):
        img_name = img_names[i]
        img_caption = img_captions[i]
        if(check_if_exists(searcher, img_caption, img_name, window_size)):
            count+=1
    
    accuracy = (count/size)*100
    return accuracy

def get_uploaded_images_names():
    return set(os.listdir(IMAGE_DIR_PATH))

if __name__ == '__main__':
    SimilarityProvider.setSimilarityCalculator(CosineSimilarity())
    Logger.mode=2
    images_uploaded = get_uploaded_images_names()
    with open(TEST_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        img_names = []
        img_captions = []
        fields = next(csvreader)
        try:
            for row in csvreader:
                if(len(row)==1):
                    first_col = row[0].split('|')
                    if(len(first_col)>=3):
                        if first_col[0].strip() in images_uploaded:
                            img_names.append(first_col[0].strip())
                            img_captions.append(first_col[2].strip())

        except Exception as e:
            Logger.e("CSV", str(e))
        accuracy = test(LinearSearcher(GoogleTextConverter()), img_names, img_captions, 10)
        Logger.mode = 3
        Logger.i("Test", str(accuracy))
    




