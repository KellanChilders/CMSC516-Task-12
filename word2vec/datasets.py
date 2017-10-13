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


class SemEvalData:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', {})
        if len(self.data) > 0:
            return

        raw_data = pd.read_csv(kwargs['file'], sep='\t').to_records()

        # Data that has been given in file.
        self.pretext = {x[1]: list(x)[5:] for x in raw_data}
        self.warrants = {x[1]: [x[2], x[3]] for x in raw_data}
        self.tags = {x[1]: x[4] for x in raw_data}

        # Data that has been generated from the pretext & warrants.
        self.p_data = {x[1]: [] for x in raw_data}
        self.w_data = {x[1]: {0: [], 1: []} for x in raw_data}

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
        # To be extended in stage2.
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


if __name__ == '__main__':
    """Simple demonstration of capabilities of the dataset."""
    import args
    dataset = SemEvalData(file=args.train_file())

    dataset.remove_stop_words()
    dataset.tag_pos()
    dataset.add_ngrams()
    print(dataset.w_data['13319707_476_A1DJNUJZN8FE7N'])
