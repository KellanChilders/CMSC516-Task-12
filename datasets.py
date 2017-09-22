import pandas as pd
import nltk
from timedmethod import timedmethod


class SemEvalData:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', {})
        if len(self.data) > 0:
            return

        # raw_data = pd.read_csv(kwargs['file'], sep='\t').to_records()
        raw_data = pd.read_csv(kwargs['file'], sep='\t')
        print(list(raw_data))
        raw_data = raw_data.to_records()
        self.data = {x[1]: [] for x in raw_data}
        print(self.data)

    def add_unigrams(self):
        pass


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


if __name__ == '__main__':
    args = reader_args()
    from os.path import join
    file = join(args.d, args.trd, args.tr)
    datasets = SemEvalData(file=file)
    # print(list(train))
    # print(datasets.train.head(1).reason[0])
