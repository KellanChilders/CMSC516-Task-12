#!/bin/sh

cd rule_based/
perl runit.sh
cd ..
echo

cd word2vec
python3 evaluator.py
