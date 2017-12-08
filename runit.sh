#!/bin/sh

cd word2vec
python3 neuralnet.py -dbg

cd ../rule_based/
perl runit.sh

cd ..
python3 vote.py word2vec/output.csv