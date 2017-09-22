import pandas as pd
from timedmethod import timedmethod


class ArgumentData:
    def __init__(self, **kwargs):
        self.train, self.test = kwargs.get('train', []), kwargs.get('test', [])

        cla = kwargs.get('cla')
        if cla is not None:
            train_raw, test_raw = self.generate_datasets(**cla)



    # @timedmethod(4)
    def generate_datasets(self, **kwargs):
        from os.path import join
        train_path = join(kwargs['d'], kwargs['trd'], kwargs['tr'])
        test_path = join(kwargs['d'], kwargs['tsd'], kwargs['ts'])

        training = pd.read_csv(train_path, sep='\t')
        testing = None if not kwargs['t'] else pd.read_csv(test_path, sep='\t')
        return training, testing

    def get_ngrams(self):
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
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = reader_args()
    datasets = ArgumentData(cla=args)
    # print(list(train))
    print(datasets.train.head(1).reason[0])
