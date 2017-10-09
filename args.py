from os.path import join

def parse_args():
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
    parser.add_argument('-t', help='Do testing',
                        type=bool, default=False)
    return parser.parse_args()


args = parse_args()


def train_file():
    return join(args.d, args.trd, args.tr)


def test_file():
    return join(args.d, args.tsd, args.ts)
