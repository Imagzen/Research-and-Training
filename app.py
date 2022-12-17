import argparse
from uploader import LocalUploader
from search_techniques.searcher import LinearSearcher
from search_techniques.searcher import GreedySearcher
from converters.image_converters import LavisImageToVectorConverter
from converters.text_converters import GoogleTextConverter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imagezen command line application")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store_true') # interactive mode
    subparser = parser.add_subparsers(dest = 'command', required=True)
    search = subparser.add_parser('search')
    upload = subparser.add_parser('upload')
    lister = subparser.add_parser('list')
    search.add_argument('--desc', type = str, help = 'Image description')
    upload.add_argument('--path', type = str, help = 'Image path')
    args = parser.parse_args()
    print("Initializing Environment")
    text_converter = GoogleTextConverter()
    image_converter = LavisImageToVectorConverter(text_converter)
    if args.command == 'search':
        searcher = LinearSearcher(text_converter)
        if args.i:
            while True:
                desc = input('>>>')
                searcher.search(desc)
        else:
            searcher.search(args.desc)
    elif args.command == 'upload':
        uploader = LocalUploader(image_converter)
        if args.i:
            while True:
                path = input('>>>')
                uploader.upload(path)
        else:
            uploader.upload(args.path)