#!/bin/sh
#Runs semeval_script
perl semeval_script.pl Semeval_Data/dev-full.txt SentiWords_1.0/SentiWords_1.0.txt
perl semeval_script.pl Semeval_Data/train-full.txt SentiWords_1.0/SentiWords_1.0.txt
