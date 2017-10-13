"""
Author      Kellan Childers
Function    Contains evaluation metrics & runs both baseline and embedder.
            While the other files contain demonstrations of their use,
            this file combines them all together.
"""
from datasets import SemEvalData


class Evaluator:
    """Standard evaluation metrics in a convenient spot."""
    def __init__(self):
        pass

    def read_csv(self, file):
        pass

    @staticmethod
    def compare(actual, predicted):
        """Generate a confusion matrix.
        Must be done before other evaluations."""
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
        """Quick way to format a 2-class confusion matrix for human reading."""
        return '{}\t{}\n{}\t{}'.format(cm[0], cm[1], cm[2], cm[3])


if __name__ == '__main__':
    """Execute the entire predictor, and display accuracy."""
    import args
    try:
        dataset = SemEvalData(file=args.train_file())
    except FileNotFoundError:
        # Couldn't find dataset, will need to specify.
        print('Could not find dataset at', args.train_file(),
              '\nPlease set the parent dir (-d), training dir (-trd),'
              'and dataset location (-tr) and execute script again.')
        raise SystemExit

    # Demonstrate baseline performance.
    print('Predicting via majority decider')
    from baseline import MajorityDecider
    maj_decider = MajorityDecider(dataset)
    maj_decider.train()

    # Evaluate baseline.
    base_predictions = maj_decider.predict(dataset)
    base_cm = Evaluator.compare(dataset.tags, base_predictions)
    print('Baseline accuracy:', round(Evaluator.accuracy(base_cm)*100, 2), '%')

    print()
    # Demonstrate word embedder performance.
    print('Predicting via word embedder')
    from embedder import WordEmbedder
    import nltk.corpus as nc
    embedder = WordEmbedder(train=nc.brown.sents())

    # Evaluate embedder.
    predictions = embedder.closest(dataset.p_data, dataset.w_data)
    confusion_matrix = Evaluator.compare(dataset.tags, predictions)
    print('Embedder accuracy:', round(Evaluator.accuracy(confusion_matrix)
                                      *100, 2), '%')
