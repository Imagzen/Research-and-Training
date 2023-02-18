import argparse
from uploader.uploader import LocalUploader
from uploader.tasks import KMeansImageAddTask
from search_techniques.searchers import LinearSearcher
from search_techniques.searchers import GreedySearcher
from search_techniques.searchers import KMeansSearching
from converters.image_converters import LavisImageToVectorConverter
from converters.text_converters import GoogleTextConverter
from config import *
from similarityfunctions.similarity import CosineSimilarity
from similarityfunctions.provider import SimilarityProvider
import os
import tracemalloc
import time
from logger.Logger import Logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == '__main__':
    SimilarityProvider.setSimilarityCalculator(CosineSimilarity())
    parser = argparse.ArgumentParser(description="Imagezen command line application")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true') # interactive mode
    parser.add_argument('-d', action='store_true') # debug mode
    subparser = parser.add_subparsers(dest = 'command', required=True)
    search = subparser.add_parser('search')
    group = search.add_mutually_exclusive_group(required=True)
    group.add_argument('--linear', action='store_true')
    group.add_argument('--quick', action='store_true')
    group.add_argument('--clustering', action = 'store_true')
    upload = subparser.add_parser('upload')
    lister = subparser.add_parser('list')
    search.add_argument('--desc', type = str, help = 'Image description')
    upload.add_argument('--path', type = str, help = 'Image path')
    args = parser.parse_args()
    tracemalloc.start()
    start_time = time.time()
    if args.d:
        Logger.mode = 4
    if args.command == 'search':
        Logger.i("app", "Initializing environment")
        text_converter = GoogleTextConverter()
        searcher = None
        if args.linear:
            searcher = LinearSearcher(text_converter)
            Logger.i("app", "Using linear Search")
        elif args.quick:
            searcher = GreedySearcher(text_converter)
            Logger.i("app", "Using Quick Search")
        elif args.clustering:
            searcher = KMeansSearching(text_converter)
            Logger.i("app", "Using Clustering Search")
        if args.i:
            while True:
                desc = input('>>>')
                if desc == 'exit':
                    break
                searcher.search(desc)
        else:
            searcher.search(args.desc)

        Logger.d("app", "Peak memory usage "+str(tracemalloc.get_traced_memory()[1])) 
        Logger.d("app", "Execution time "+str(time.time() - start_time))
        
    elif args.command == 'upload':
        Logger.i("app", "Initializing environment")
        text_converter = GoogleTextConverter()
        image_converter = LavisImageToVectorConverter(text_converter)
        uploader = LocalUploader(image_converter, [KMeansImageAddTask()])
        if args.i:
            while True:
                path = input('>>>')
                if path == 'exit':
                    break
                uploader.upload(path)
        else:
            uploader.upload(args.path)

        Logger.d("app", "Peak memory usage "+str(tracemalloc.get_traced_memory()[1])) 
        Logger.d("app", "Execution time "+str(time.time() - start_time))
    elif args.command == 'list':
        files = os.listdir(IMAGE_DIR_PATH)
        for f in files:
            Logger.i("File", f)
        Logger.i("app", str(len(files))+" images found.")
    
    tracemalloc.stop()