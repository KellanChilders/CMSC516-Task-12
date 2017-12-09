"""
Author      Kellan Childers
Function    Helps parse command line arguments quickly.
            When imported, allows a program to take cla, or show help with -h.
"""
from os.path import join, abspath


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
    parser.add_argument('-tsd', help='Testing directory',
                        type=str, default='test')
    parser.add_argument('-g', help='GoogleNews corpus',
                        type=str,
                        default='GoogleNews-vectors-negative300.bin.gz')
    parser.add_argument('-c', help='Write to csv',
                        type=str, default='word2vec.csv')
    parser.add_argument('-dbg', help='Debug mode',
                        action='store_true', default=False)
    parser.add_argument('-t', help='Do testing',
                        type=bool, default=False)
    return parser.parse_args()


args = parse_args()


def train_file():
    """Get the full path to the training dataset (os agnostic)."""
    return join(args.d, args.trd, args.tr)


def test_file():
    """Get the full path to the testing dataset (os agnostic)."""
    return join(args.d, args.tsd, args.ts)


def google_file():
    """Get the file name of the google corpus."""
    return args.g


def csv_file():
    """Get the full path to the csv file (os agnostic)."""
    return abspath(args.c)


def debug():
    """Tag for neural network to use debug datasets."""
    return args.dbg
