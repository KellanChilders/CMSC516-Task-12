"""
Author    Andrew Ward & Kellan Childers
Function  Voting script.  Take confidence measures from any number of
          input CSV files, creates one dictionary containing only the most
          confident predictions.
          Dict format: Key=debate id.  Value=(prediction, confidence)
          Usage: python3 vote.py [filenames]
          CSV format: debate id, prediction, confidence
"""

import csv


class Voter:
    def __init__(self):
        self.votes={}

    def vote(self, *filenames):
        """Evaluate most likely choice from csv files."""
        for file in filenames:
            csvfile = open(file, newline='')
            confid = csv.reader(csvfile)
            for row in confid:
                if ((row[0] not in self.votes) or
                        (float(row[2]) > self.votes[row[0]][1])):
                    self.votes[row[0]]=(int(row[1]),float(row[2]))
            csvfile.close()

    def display(self):
        """Print out results of vote."""
        for k,v in self.votes.items():
            print("Debate ID",k,": ", "Prediction = ",v[0],
                  ", Confidence = ",v[1],sep="")


if __name__ == '__main__':
    import sys
    from os.path import join

    voter = Voter()
    voter.vote(*sys.argv[1:])
    # voter.display()
    # print(voter.votes)

    # Load dataset for actual tags.
    from word2vec.datasets import SemEvalData
    dataset = SemEvalData(file=join('word2vec', 'data',
                                    'train', 'train-full.txt'))

    # Evaluate results of voting.
    from word2vec.evaluator import Evaluator
    results = Evaluator.simplify(voter.votes)
    tags = dataset.tags

    # Output accuracy.
    confusion_matrix = Evaluator.compare(tags, results)
    print('Overall accuracy:', round(Evaluator.accuracy(confusion_matrix)
                                     *100, 2), '%')


