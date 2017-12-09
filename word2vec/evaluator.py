"""
Author      Kellan Childers
Function    Contains evaluation metrics & runs both baseline and embedder.
            While the other files contain demonstrations of their use,
            this file combines them all together.
"""


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

    @staticmethod
    def simplify(pred):
        """Reduce prediction into manageable chunk."""
        return {key: val[0] for key, val in pred.items()}


if __name__ == '__main__':
    """Execute the entire predictor, and display accuracy."""
    import args
    from datasets import SemEvalData

    try:
        dataset = SemEvalData(file=args.train_file())
    except FileNotFoundError:
        # Couldn't find dataset, will need to specify.
        print('Could not find dataset at', args.train_file(),
              '\nPlease set the parent dir (-d), training dir (-trd),'
              'and dataset location (-tr) and execute script again.')
        raise SystemExit

    dataset.expand_contraction()
    dataset.remove_common()
    dataset.add_bag_words()

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
    print('Loading the word embedder')
    from embedder import WordEmbedder
    embedder = WordEmbedder(load=args.google_file())

    print('Generating similarity measures')
    # Evaluate embedder.
    embed_predictions = embedder.raw_closest(dataset.p_data, dataset.w_data)
    embed_cm = Evaluator.compare(dataset.tags, embed_predictions)
    print('Similarity accuracy:', round(Evaluator.accuracy(embed_cm)
                                        * 100, 2), '%')

    print()
    # Demonstrate neural network performance.
    print('Predicting via word embedder and neural network',
          'using 10 fold cross validation')
    from neuralnet import NeuralNet

    # Create 10 folds for cross validation.
    num_folds = 10
    datasets, orders = dataset.datasets_from_folds(num_folds)

    accuracies = []
    network_predictions = {}
    for i in range(num_folds):
        # Create a dataset of the non-test datasets.
        ds = SemEvalData(blank=True)
        for j in range(num_folds):
            if j != i:
                ds = ds + datasets[j]

        input, tags, order = NeuralNet.format_dataset(ds, embedder)

        # Train the network.
        network = NeuralNet(hidden=128, layers=2, input=len(input[0]), output=2)

        # Train on non-test datasets.
        loss, accuracy = network.train(input, tags, iterations=10)
        accuracies += [accuracy]

        # Test on test dataset.
        test, _, order = NeuralNet.format_dataset(datasets[i], embedder)
        network_predictions.update(network.predict(test, order))

    # Take the average of the accuracies for overall accuracy.
    accuracy = sum(accuracies) / len(accuracies)
    print("Neural Network Accuracy: " + str(round(accuracy*100, 2)) + "%")

    # Save similarity measures to csv.
    with open(args.csv_file(), 'w') as writefile:
        writefile.write(embedder.to_csv(embed_predictions))

    # Save neural network to csv.
    with open('output.csv', 'w') as writefile:
        writefile.write(NeuralNet.pred_to_csv(network_predictions))
