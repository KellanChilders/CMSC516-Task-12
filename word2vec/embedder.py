"""
Author      Kellan Childers
Function    Contains the word2vec implementation of our algorithm.
            The WordEmbedder class trains via a corpus, then is predicts
            the warrant which is most related to the claim.
"""
import math
from functools import reduce
import nltk.corpus as nc
import gensim.models as gs


class WordEmbedder:
    """Predict warrants based on word2vec similarity."""
    def __init__(self, **kwargs):
        load_file = kwargs.get('load', None)
        if load_file is not None:
            # self.model = gs.Word2Vec.load(load_file)
            self.model = gs.KeyedVectors.load_word2vec_format(load_file,
                                                              binary=True)
        else:
            training_corpora = kwargs.get('train', nc.brown.sents())
            min_count = kwargs.get('count', 1)
            iterations = kwargs.get('iter', 5)
            workers = kwargs.get('workers', 4)
            self.model = gs.Word2Vec(training_corpora, workers=workers,
                                     min_count=min_count, iter=iterations)

        save_file = kwargs.get('save', None)
        if save_file is not None:
            self.save(save_file)

        # Need one word we know is in corpus so we can create n-len vectors.
        baseline = kwargs.get('base', 'i')
        self.vec_length = len(self.model[baseline])

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = gs.Word2Vec.load(filename)

    def get_vector(self, sent):
        """Get the vector representations of every word in a sentence."""
        if type(sent) is str:
            sent = sent.split()
        return [[0 for _ in range(self.vec_length)] if word not in self.model
                else self.model[word] for word in sent]

    @staticmethod
    def distance(vector):
        """Calculate euclidean distance of a vector."""
        return math.sqrt(sum(v**2 for v in vector))

    def sum_normalize(self, vector):
        """Sum every vector in a sentence, then normalize to distance 1."""
        if len(vector) == 0:
            return [0 for _ in range(self.vec_length)]

        vec = [reduce(lambda x, y: x + y, (v[i] for v in vector))
               for i in range(self.vec_length)]
        mag = self.distance(vec)

        if mag == 0:
            return [0 for _ in range(self.vec_length)]

        return [x/mag for x in vec]

    def process(self, pretext, warrants):
        """Find vectors, sum, and normalize all elements of a dataset."""
        # Find word embeddings of warrants & claim.
        base = {key: self.get_vector(sent) for key, sent in pretext.items()}
        claim = {key: {0: self.get_vector(sent[0]),
                       1: self.get_vector(sent[1])}
                 for key, sent in warrants.items()}

        # Condense word vectors into normalize sentence vector.
        base = {key: self.sum_normalize(val) for key, val in base.items()}
        claim = {key: {0: self.sum_normalize(val[0]),
                       1: self.sum_normalize(val[1])}
                 for key, val in claim.items()}

        return base, claim

    def similarity(self, vec1, vec2):
        """Calculate the cosine similarity between two vectors."""
        if sum(vec1) == 0 or sum(vec2) == 0:
            return 0
        dot = sum(x*y for x, y in zip(vec1, vec2))

        return dot / (self.distance(vec1)*self.distance(vec2))

    @staticmethod
    def confidence(comparisons):
        """Get the confidence of an answer."""
        greater = {key: max(val.values()) for key, val in comparisons.items()}
        lesser = {key: min(val.values()) for key, val in comparisons.items()}
        difference = {key: (greater[key] - lesser[key])/2 + .5
                      for key in greater.keys()}

        return difference

    def closest(self, pretext, warrants):
        """Predict the closest warrant to the claim."""
        base, claim = self.process(pretext, warrants)

        # Find the cosine similarity between claim and warrants.
        compare = {key: {0: self.similarity(sent, claim[key][0]),
                         1: self.similarity(sent, claim[key][1])}
                   for key, sent in base.items()}
        confidence = self.confidence(compare)

        # Pick most likely, defaulting to warrant1.
        return {key: [0 if item[0] > item[1] else 1, confidence[key]]
                for key, item in compare.items()}

    def raw_closest(self, pretext, warrants):
        """Predict the closest warrant to the claim."""
        base, claim = self.process(pretext, warrants)

        # Find the cosine similarity between claim and warrants.
        compare = {key: {0: self.similarity(sent, claim[key][0]),
                         1: self.similarity(sent, claim[key][1])}
                   for key, sent in base.items()}
        confidence = self.confidence(compare)

        # Pick most likely, defaulting to warrant1.
        return {key: 0 if item[0] > item[1] else 1
                for key, item in compare.items()}

    @staticmethod
    def to_csv(prediction):
        """Format into csv for storing results."""
        return '\n'.join('{},{},{}'.format(key, value[0], value[1])
                         for key, value in prediction.items())


if __name__ == '__main__':
    """Simple demonstration of embedder."""
    import args
    from datasets import SemEvalData

    dataset = SemEvalData(file=args.train_file())

    # Some preprocessing steps for dataset.
    dataset.expand_contraction()
    dataset.remove_common()
    dataset.add_bag_words()

    # Train, predict, and show as csv.
    embedder = WordEmbedder(load=args.google_file())
    predictions = embedder.raw_closest(dataset.p_data, dataset.w_data)

    # Save predictions to csv.
    with open('embed_output.csv', 'w') as writefile:
        writefile.write(embedder.to_csv(predictions))
