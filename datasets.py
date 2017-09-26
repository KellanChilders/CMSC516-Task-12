import pandas as pd
import nltk
from timedmethod import timedmethod


class SemEvalData:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', {})
        if len(self.data) > 0:
            return

        raw_data = pd.read_csv(kwargs['file'], sep='\t').to_records()
        self.pretext = {x[1]: list(x)[5:] for x in raw_data}
        self.warrants = {x[1]: [x[2], x[3]] for x in raw_data}
        self.tags = {x[1]: x[4] for x in raw_data}

        # Data that has been generated from the pretext & warrants.
        self.p_data = {x[1]: [] for x in raw_data}
        self.w_data = {x[1]: [] for x in raw_data}

    def add_ngrams(self):
        for key, value in self.data.items():
            # nltk.word_tokenize(text)
            # print(key, value)
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
    dataset = SemEvalData(file=file)
    # print(list(train))
    # print(datasets.train.head(1).reason[0])
