""" compare output.csv tags with the train-full.txt data """
""" author: wardac """

import args
import pandas as pd

raw_data = pd.read_csv(args.train_file(), sep='\t').to_records()
tags = {x[1]: x[4] for x in raw_data}
pred_data = pd.read_csv('output.csv', sep=',').to_records()
preds = {x[1]: x[2] for x in pred_data}

correct = 0
for key in tags.keys():
    if key in preds:
        # print(key,'\t',tags[key],'\t',preds[key])
        if tags[key] == preds[key]:
            correct = correct + 1
    else:
        print(key,'key not found')

print('accuracy',correct/len(tags))
lost = len(tags)-len(preds)
if lost: print(lost,'key(s) lost')
