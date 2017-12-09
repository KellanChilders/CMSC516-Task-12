import numpy as np
import keras.models as km
import keras.layers as kl
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
                         optimizer='sgd', metrics=['accuracy'])

    def train(self, training, tags, iterations, verbose=0):
        self.net.fit(training, tags, epochs=iterations, verbose=verbose)
        return self.net.evaluate(training, tags)

    def predict(self, testing, order):
        pred = self.net.predict(testing)
        return {o: [0 if pred[i][0] > pred[i][1] else 1,
                    pred[i][0] if pred[i][0] > pred[i][1] else pred[i][1]]
                for o, i in zip(order, range(len(pred)))}

    @staticmethod
    def pred_to_csv(pred):
        return '\n'.join(','.join([key, str(val[0]), str(val[1])])
                         for key, val in pred.items())

    @staticmethod
    def format_dataset(data, embed):
        order = sorted(data.tags.keys())
        base, claim = embed.process(data.p_data, data.w_data)

        input = np.array(list(base[i] + claim[i][0] + claim[i][1]
                              for i in order))

        tags = np.array([data.tags[i] for i in order])
        tags = np.array([.9999 if i == 1 else 0 for i in tags])
        # tags.reshape((-1, 2))
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

    if args.debug():
        input = np.genfromtxt('input.csv', delimiter=',')
        tags = np.genfromtxt('tags.csv')
        order = sorted(dataset.tags.keys())
    else:

        # Some preprocessing steps for dataset.
        dataset.expand_contraction()
        dataset.remove_common()
        dataset.add_bag_words()

        # Train, predict, and show as csv.
        embedder = WordEmbedder(load=args.google_file())
        input, tags, order = NeuralNet.format_dataset(dataset, embedder)

    network = NeuralNet(input=len(input[0]), output=2)
    loss, accuracy = network.train(input, tags, iterations=10)
    print()
    print("Accuracy: " + str(round(accuracy*100, 2)) + "%")
    print()
    predictions = network.predict(input, order)

    with open('output.csv', 'w') as writefile:
        writefile.write(network.pred_to_csv(predictions))

