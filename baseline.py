from datasets import SemEvalData


class MajorityDecider:
    def __init__(self, data):
        self.data = data
        self.rtn = 0

    def train(self):
        zero = list(self.data.tags.values()).count('0')
        one = list(self.data.tags.values()).count('1')
        self.rtn = 0 if zero > one else 1

    def predict(self, sample):
        return {item: self.rtn for item in sample.w_data}

    @staticmethod
    def to_csv(prediction):
        return '\n'.join('{}\t{}'.format(key, value)
                         for key, value in prediction.items())


if __name__ == '__main__':
    import args
    dataset = SemEvalData(file=args.train_file())

    decider = MajorityDecider(dataset)
    decider.train()

    print(decider.to_csv(decider.predict(dataset)))
