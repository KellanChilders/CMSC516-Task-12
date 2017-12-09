#!/bin/sh

cd rule_based/
perl runit.sh
cd ..
echo

cd word2vec
python3 evaluator.py

cd ..
python3 vote.py word2vec/output.csv word2vec/embed_output.csv rule_based/output.csv