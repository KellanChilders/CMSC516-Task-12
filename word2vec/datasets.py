"""
Author      Kellan Childers
Function    Creates an easily-modifiable dataset and offers common
            preprocessing. It reads in a dataset, then formats it according
            to the task 12 structure, and initializes a feature set for each
            claim and warrant.
"""
import pandas as pd
import nltk
import re
from random import shuffle
from math import floor


class SemEvalData:
    def __init__(self, **kwargs):
        if kwargs.get('blank', False) is True:
            self.__raw()
            return

        raw_data = pd.read_csv(kwargs['file'], sep='\t').to_records()

        # Data that has been given in file.
        self.pretext = {x[1]: list(x)[5:] for x in raw_data}
        self.warrants = {x[1]: [x[2], x[3]] for x in raw_data}
        self.tags = {x[1]: x[4] for x in raw_data}

        # Data that has been generated from the pretext & warrants.
        self.p_data = {x[1]: [] for x in raw_data}
        self.w_data = {x[1]: {0: [], 1: []} for x in raw_data}

    def __raw(self):
        """Reset all dictionaries."""
        self.pretext = {}
        self.warrants = {}
        self.tags = {}
        self.p_data = {}
        self.w_data = {}

    def folds(self, num_folds):
        """Create even folds."""
        ids = list(self.tags.keys())

        count = len(ids)
        fold_size, additional = floor(count/num_folds), count % num_folds

        order = list(range(count))
        shuffle(order)

        fold, orders = [], []
        for _ in range(num_folds):
            fold += [[ids[i] for i in order[:fold_size]]]
            orders += [order[:fold_size]]
            order = order[fold_size:]

        for i in range(additional):
            fold[i] += [ids[order[0]]]
            orders[i] += [order[0]]
            order = order[1:]

        return fold, orders

    def datasets_from_folds(self, num_folds):
        """Create datasets from a set of folds."""
        folds, orders = self.folds(num_folds)
        datasets = []

        for fold in folds:
            ds = SemEvalData(blank=True)
            ds.pretext = {key: value for key, value in self.pretext.items()
                          if key in fold}
            ds.warrants = {key: value for key, value in self.warrants.items()
                           if key in fold}
            ds.tags = {key: value for key, value in self.tags.items()
                       if key in fold}
            ds.p_data = {key: value for key, value in self.p_data.items()
                         if key in fold}
            ds.w_data = {key: value for key, value in self.w_data.items()
                         if key in fold}
            datasets += [ds]

        return datasets, orders

    def remove_stop_words(self, stop=('a', 'an', 'the', 'to')):
        """Remove common words from pretext and warrants."""
        for key, value in self.pretext.items():
            for i, item in enumerate(value):
                tokens = nltk.word_tokenize(item)
                tokens = [x for x in tokens if x not in stop]
                value[i] = ' '.join(tokens)
            self.pretext[key] = value

        for key, value in self.warrants.items():
            for i, item in enumerate(value):
                tokens = nltk.word_tokenize(item)
                tokens = [x for x in tokens if x not in stop]
                value[i] = ' '.join(tokens)
            self.warrants[key] = value

    def expand_contraction(self):
        """Replace n't with not."""
        contractions = {"n't": ' not '}
        for key, value in self.pretext.items():
            for i, item in enumerate(value):
                item = re.sub(r'n\'t ', ' not ', item)
                value[i] = item
            self.pretext[key] = value

        for key, value in self.warrants.items():
            for i, item in enumerate(value):
                item = re.sub(r'n\'t ', ' not ', item)
                value[i] = item
            self.warrants[key] = value

    def tag_pos(self):
        """Tag each word in pretext and warrants."""
        for key, value in self.pretext.items():
            for i, item in enumerate(value):
                tokens = nltk.word_tokenize(item)
                tags = ' '.join('/'.join(x) for x in nltk.pos_tag(tokens))
                value[i] = tags
            self.pretext[key] = value

        for key, value in self.warrants.items():
            for i, item in enumerate(value):
                tokens = nltk.word_tokenize(item)
                tags = ' '.join('/'.join(x) for x in nltk.pos_tag(tokens))
                value[i] = tags
            self.warrants[key] = value

    def add_bag_words(self):
        """Add only unigrams to the pretext and each warrant."""
        for key, value in self.pretext.items():
            unigrams = [nltk.word_tokenize(sent) for sent in value]

            unigrams = [x for i in unigrams for x in i]

            self.p_data[key] += unigrams

        for key, value in self.warrants.items():
            unigrams = [nltk.word_tokenize(sent) for sent in value]

            self.w_data[key][0] += unigrams[0]

            self.w_data[key][1] += unigrams[1]

    def add_ngrams(self):
        """Add unigrams and bigrams to the pretext and each warrant."""
        for key, value in self.pretext.items():
            unigrams = [nltk.word_tokenize(sent) for sent in value]
            bigrams = [list(nltk.ngrams(sent, 2)) for sent in unigrams]

            unigrams = [x for i in unigrams for x in i]
            bigrams = [x for i in bigrams for x in i]

            self.p_data[key] += unigrams
            self.p_data[key] += bigrams

        for key, value in self.warrants.items():
            unigrams = [nltk.word_tokenize(sent) for sent in value]
            bigrams = [list(nltk.ngrams(sent, 2)) for sent in unigrams]

            self.w_data[key][0] += unigrams[0]
            self.w_data[key][0] += bigrams[0]

            self.w_data[key][1] += unigrams[1]
            self.w_data[key][1] += bigrams[1]

    def remove_common(self):
        """Remove words each claim shares."""
        self.w_data = {key: {0: [x for x in val[0] if x not in val[1]],
                             1: [x for x in val[1] if x not in val[0]]}
                       for key, val in self.w_data.items()}

    def __add__(self, other):
        ds = SemEvalData(blank=True)

        pretext = self.pretext.copy()
        pretext.update(other.pretext)
        ds.pretext = pretext

        warrants = self.warrants.copy()
        warrants.update(other.warrants)
        ds.warrants = warrants

        tags = self.tags.copy()
        tags.update(other.tags)
        ds.tags = tags

        p_data = self.p_data.copy()
        p_data.update(other.p_data)
        ds.p_data = p_data

        w_data = self.w_data.copy()
        w_data.update(other.w_data)
        ds.w_data = w_data

        return ds


if __name__ == '__main__':
    """Simple demonstration of capabilities of the dataset."""
    import args
    dataset = SemEvalData(file=args.train_file())

    dataset.remove_stop_words()
    dataset.tag_pos()
    dataset.add_ngrams()
    # print(dataset.w_data['13319707_476_A1DJNUJZN8FE7N'])

    # print(len(dataset.pretext))
    # print('\n'.join(str(x) for x in dataset.folds(9)))

    folded_datasets, orders = dataset.datasets_from_folds(10)
    test = folded_datasets[0] + folded_datasets[1]
    # print('\n'.join(str(len(o)) for o in orders))
    print(list(sorted(folded_datasets[0].tags.keys())))
    print(list(sorted(folded_datasets[1].tags.keys())))
    print(list(sorted(test.tags.keys())))
