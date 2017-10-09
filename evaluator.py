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
    def format_conf_mat(cm):
        return '{}\t{}\n{}\t{}'.format(cm[0], cm[1], cm[2], cm[3])


if __name__ == '__main__':
    import args
    dataset = SemEvalData(file=args.train_file())

    from baseline import MajorityDecider
    decider = MajorityDecider(dataset)
    decider.train()

    predictions = decider.predict(dataset)

    confusion_matrix = Evaluator.compare(dataset.tags, predictions)
    print(Evaluator.format_conf_mat(confusion_matrix))
