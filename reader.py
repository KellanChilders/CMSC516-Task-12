import pandas as pd
from timedmethod import timedmethod


def reader_args():
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


@timedmethod(4)
def generate_datasets(cla):
    from os.path import join
    training = pd.read_csv(join(cla.d, cla.trd, cla.tr), sep='\t')
    testing = None if not cla.t else \
        pd.read_csv(join(cla.d, cla.tsd, cla.ts), sep='\t')
    return training, testing


if __name__ == '__main__':
    args = reader_args()
    train, test = generate_datasets(args)
    print(list(train))
