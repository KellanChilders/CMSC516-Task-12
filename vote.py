# Voting script.  Take confidence measures from any number of input CSV files,
# creates one dictionary containing only the most confident predictions.
# Dict format: Key=debate id.  Value=(prediction, confidence)
# Usage: python3 vote.py [filenames] 
#    CSV format: debate id, prediction, confidence
# - wardac

import csv


class Voter:
    def __init__(self):
        self.votes={}

    def vote(self, *filenames):
        for idx, fn in enumerate(filenames):
            csvfile = open(fn, newline='')
            confid = csv.reader(csvfile)
            for row in confid:
                if ((row[0] not in self.votes) or
                        (float(row[2]) > self.votes[row[0]][1])):
                    self.votes[row[0]]=(int(row[1]),float(row[2]))
            csvfile.close()

    def display(self):
        for k,v in self.votes.items():
            print("Debate ID",k,": ", "Prediction = ",v[0],
                  ", Confidence = ",v[1],sep="")


if __name__ == '__main__':
    import sys
    voter = Voter()
    voter.vote(*sys.argv[1:])
    # voter.display()
    # print(voter.votes)

    import numpy as np
    from word2vec.evaluator import Evaluator
    results = Evaluator.simplify(voter.votes)
    order = np.genfromtxt('word2vec/order.csv', dtype='|U16')
    tags = np.genfromtxt('word2vec/tags.csv')
    tags = {o: t for o, t in zip(order, tags)}

    confusion_matrix = Evaluator.compare(tags, results)
    print('Voter accuracy:', round(Evaluator.accuracy(confusion_matrix)
                                      *100, 2), '%')


