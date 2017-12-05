"""
Author      Kellan Childers
Function    Helps parse command line arguments quickly.
            When imported, allows a program to take cla, or show help with -h.
"""
from os.path import join


def parse_args():
    """Read in arguments to help other programs."""
    import argparse as ap

    parser = ap.ArgumentParser(
        description='Train an evaluator to decide between arguments.')
    parser.add_argument('-d', help='Datasets directory',
                        type=str, default='data')
    parser.add_argument('-tr', help='Training dataset',
                        type=str, default='train-full.txt')
    parser.add_argument('-trd', help='Training directory',
                        type=str, default='train')
    parser.add_argument('-ts', help='Testing dataset',
                        type=str, default='test.tsv')
    parser.add_argument('-w', help='Wikipedia corpus',
                        type=str, default='enwiki-20170820-pages-articles.xml.bz2')
    parser.add_argument('-tsd', help='Testing directory',
                        type=str, default='test')
    parser.add_argument('-t', help='Do testing',
                        type=bool, default=False)
    parser.add_argument('-e', help='Create wiki corpus',
                        type=bool, default=False)
    return parser.parse_args()


args = parse_args()


def train_file():
    """Get the full path to the training dataset (os agnostic)."""
    return join(args.d, args.trd, args.tr)


def test_file():
    """Get the full path to the testing dataset (os agnostic)."""
    return join(args.d, args.tsd, args.ts)


def wiki():
    """Get the location of the Wikipedia corpus."""
    return args.w
