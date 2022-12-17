import argparse
from uploader.uploader import LocalUploader
from search_techniques.searchers import LinearSearcher
from search_techniques.searchers import GreedySearcher
from converters.image_converters import LavisImageToVectorConverter
from converters.text_converters import GoogleTextConverter
from config import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imagezen command line application")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true') # interactive mode
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
    if args.command == 'search':
        print("Initializing Environment")
        text_converter = GoogleTextConverter()
        searcher = None
        if args.linear:
            searcher = LinearSearcher(text_converter)
            print('Using linear search')
        elif args.quick:
            searcher = GreedySearcher(text_converter)
            print('Using quick search')
        if args.i:
            while True:
                desc = input('>>>')
                searcher.search(desc)
        else:
            searcher.search(args.desc)
    elif args.command == 'upload':
        print("Initializing Environment")
        text_converter = GoogleTextConverter()
        image_converter = LavisImageToVectorConverter(text_converter)
        uploader = LocalUploader(image_converter, [])
        if args.i:
            while True:
                path = input('>>>')
                uploader.upload(path)
        else:
            uploader.upload(args.path)
    elif args.command == 'list':
        files = os.listdir(IMAGE_DIR_PATH)
        for f in files:
            print(f)
        print(str(len(files))+" images found.")