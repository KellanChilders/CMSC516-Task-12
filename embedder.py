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

    def get_vector(self, sent):
        if type(sent) is not list:
            sent = sent.split()
        return [self.model[word] for word in sent]


if __name__ == '__main__':
    import args
    dataset = SemEvalData(file=args.train_file())
    dataset.add_bag_words()

    embedder = WordEmbedder()

    test = embedder.get_vector(dataset.w_data['13319707_476_A1DJNUJZN8FE7N'][0])
    print(dataset.w_data['13319707_476_A1DJNUJZN8FE7N'][0])
    print(test)
