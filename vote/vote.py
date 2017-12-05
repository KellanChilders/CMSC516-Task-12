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
    voter.display()
    print(voter.votes)
    
