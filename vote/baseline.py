"""
Author      Kellan Childers
Function    Creates a baseline predictor by looking solely at most likely tag.
            We can use this to then judge other predictors.
"""
from datasets import SemEvalData


class MajorityDecider:
    def __init__(self, data):
        self.data = data
        self.rtn = 0

    def train(self):
        """Look for the most common tag by counting 1 or 0."""
        zero = list(self.data.tags.values()).count('0')
        one = list(self.data.tags.values()).count('1')
        self.rtn = 0 if zero > one else 1

    def predict(self, sample):
        """Predict the tag by saying the most likely for all."""
        return {item: self.rtn for item in sample.w_data}

    @staticmethod
    def to_csv(prediction):
        """Format into csv for storing results."""
        return '\n'.join('{}\t{}'.format(key, value)
                         for key, value in prediction.items())


if __name__ == '__main__':
    """Simple demonstration of baseline."""
    import args
    dataset = SemEvalData(file=args.train_file())

    decider = MajorityDecider(dataset)
    decider.train()

    print(decider.to_csv(decider.predict(dataset)))
