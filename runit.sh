#!/bin/sh

cd word2vec
python3 neuralnet.py
python3 embedder.py

cd ../rule_based/
perl runit.sh

cd ..
python3 vote.py word2vec/output.csv word2vec/embed_output.csv