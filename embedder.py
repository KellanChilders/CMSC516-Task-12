import math
import nltk
import nltk.corpus as nc
import gensim.models as gs
from datasets import SemEvalData


class WordEmbedder:
    def __init__(self, **kwargs):
        training_corpora = kwargs.get('train', nc.brown.sents())
        min_count = kwargs.get('count', 1)
        iterations = kwargs.get('iter', 5)
        self.model = gs.Word2Vec(training_corpora,
                                 min_count=min_count, iter=iterations)
        # Need one word we know is in corpus so we can create n-len vectors.
        baseline = kwargs.get('base', 'i')
        self.vec_length = len(self.model[baseline])

    def get_vector(self, sent):
        if type(sent) is str:
            sent = sent.split()
        return [[0 for _ in range(self.vec_length)] if word not in self.model
                else self.model[word] for word in sent]

    @staticmethod
    def similarity(vec1, vec2):
        return sum(x*y for x, y in zip(vec1, vec2)) /\
               (math.sqrt(sum(x**2 for x in vec1))
                *math.sqrt(sum(x**2 for x in vec2)))

    def closest(self, pretext, warrants):
        base = {key: self.get_vector(sent) for key, sent in pretext.items()}
        claim = {key: {0: self.get_vector(sent[0]), 1: self.get_vector(sent[1])}
                 for key, sent in warrants.items()}

        compare = {key: {0: self.similarity(sent, claim[key][0]),
                         1: self.similarity(sent, claim[key][0])}
                   for key, sent in base.items()}
        return {key: 0 if item[0] > item[1] else 1
                for key, item in compare.items()}


if __name__ == '__main__':
    import args
    dataset = SemEvalData(file=args.train_file())
    dataset.add_bag_words()

    embedder = WordEmbedder()
    predictions = embedder.closest(dataset.p_data, dataset.w_data)
    print(predictions)

    example = '13319707_476_A1DJNUJZN8FE7N'
    # print(dataset.w_data[example][0])
    # print(embedder.get_vector(dataset.w_data[example][0]))
