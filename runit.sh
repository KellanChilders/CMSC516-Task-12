#!/bin/sh

cd word2vec
python3 evaluator.py

cd ../rule_based/
perl runit.sh

cd ..
python3 vote.py word2vec/output.csv rule_based/output.csv