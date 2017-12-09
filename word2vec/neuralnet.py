"""
Author      Kellan Childers
Function    Contains a simple neural network implementation via Keras.
            The neural net uses the embeddings of the warrants and pretext
            to determine the warrant that supports the pretext.
"""

import numpy as np
import keras.models as km
import keras.layers as kl
import keras.utils as ku
from datasets import SemEvalData
from embedder import WordEmbedder


class NeuralNet:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', .1)
        input_dim = kwargs.get('input', 400)
        output_dim = kwargs.get('output', 2)
        hidden = kwargs.get('hidden', 100)
        layers = kwargs.get('layers', 2)

        self.net = km.Sequential()

        self.net.add(kl.Dense(units=hidden, activation='relu',
                              input_dim=input_dim))

        for _ in range(layers-1):
            self.net.add(kl.Dense(units=hidden, activation='relu'))

        self.net.add(kl.Dense(units=output_dim, activation='softmax'))

        self.net.compile(loss='sparse_categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])

    def train(self, training, tags, iterations, verbose=0):
        """Train on a dataset."""
        self.net.fit(training, tags, epochs=iterations, verbose=verbose)
        return self.net.evaluate(training, tags)

    def predict(self, testing, order):
        """Predict a dataset."""
        pred = self.net.predict(testing)
        return {o: [0 if pred[i][0] > pred[i][1] else 1,
                    pred[i][0] if pred[i][0] > pred[i][1] else pred[i][1]]
                for o, i in zip(order, range(len(pred)))}

    @staticmethod
    def pred_to_csv(pred):
        """Create a csv file for a set of predictions."""
        return '\n'.join(','.join([key, str(val[0]), str(val[1])])
                         for key, val in pred.items())

    @staticmethod
    def format_dataset(data, embed):
        """Create a feature set the network can interpet."""
        order = sorted(data.tags.keys())
        base, claim = embed.process(data.p_data, data.w_data)

        input = np.array(list(base[i] + claim[i][0] + claim[i][1]
                              for i in order))

        tags = np.array([data.tags[i] for i in order])
        # tags = ku.to_categorical(tags, 2)
        return input, tags, order


if __name__ == "__main__":
    """Sample demo of neural net"""
    import args
    try:
        dataset = SemEvalData(file=args.train_file())
    except FileNotFoundError:
        # Couldn't find dataset, will need to specify.
        print('Could not find dataset at', args.train_file(),
              '\nPlease set the parent dir (-d), training dir (-trd),'
              'and dataset location (-tr) and execute script again.')
        raise SystemExit

    # Some preprocessing steps for dataset.
    dataset.expand_contraction()
    dataset.remove_common()
    dataset.add_bag_words()

    # Train, predict, and show as csv.
    embedder = WordEmbedder(load=args.google_file())

    # Create 10 folds for cross validation.
    num_folds = 10
    datasets, orders = dataset.datasets_from_folds(num_folds)

    accuracies = []
    predictions = {}
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
        predictions.update(network.predict(test, order))

    # Take the average of the accuracies for overall accuracy.
    accuracy = sum(accuracies)/len(accuracies)
    print("Accuracy: " + str(round(accuracy*100, 2)) + "%")

    # Save the networks results.
    with open('output.csv', 'w') as writefile:
        writefile.write(NeuralNet.pred_to_csv(predictions))

