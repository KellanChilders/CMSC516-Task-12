from datasets import SemEvalData


class Evaluator:
    def __init__(self):
        pass

    def read_csv(self, file):
        pass

    @staticmethod
    def compare(actual, predicted):
        tp = len([1 for key, act in actual.items()
                  if act == predicted[key] == 0])
        fp = len([1 for key, act in actual.items()
                  if act != predicted[key] and predicted[key] == 0])
        fn = len([1 for key, act in actual.items()
                  if act != predicted[key] and predicted[key] == 1])
        tn = len([1 for key, act in actual.items()
                  if act == predicted[key] == 1])
        return [tp, fp, fn, tn]

    @staticmethod
    def accuracy(cm):
        return (cm[0]+cm[3])/sum(cm)

    @staticmethod
    def precision(cm):
        return cm[0]/(cm[0]+cm[1])

    @staticmethod
    def recall(cm):
        return cm[0]/(cm[0]+cm[2])

    @staticmethod
    def format_conf_mat(cm):
        return '{}\t{}\n{}\t{}'.format(cm[0], cm[1], cm[2], cm[3])


if __name__ == '__main__':
    import args
    dataset = SemEvalData(file=args.train_file())

    print('Predicting via majority decider')
    from baseline import MajorityDecider
    maj_decider = MajorityDecider(dataset)
    maj_decider.train()

    base_predictions = maj_decider.predict(dataset)
    base_cm = Evaluator.compare(dataset.tags, base_predictions)
    print('Baseline accuracy:', round(Evaluator.accuracy(base_cm)*100, 2), '%')

    print()
    print('Predicting via word embedder')
    from embedder import WordEmbedder
    import nltk.corpus as nc
    embedder = WordEmbedder(train=nc.brown.sents())

    predictions = embedder.closest(dataset.p_data, dataset.w_data)
    confusion_matrix = Evaluator.compare(dataset.tags, predictions)
    print('Embedder accuracy:', round(Evaluator.accuracy(confusion_matrix)
                                      *100, 2), '%')
