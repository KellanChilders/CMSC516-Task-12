README DOCUMENTATION:

AUTHORS: Kellan Childers, Megan Davis, Andrew Ward

ROLES

Megan Davis - Author of semeval_script.pl, and procedural implementation within powerpoint. Set up trello board.
Andrew Ward - Author of task information/introduction and proposed approach in powerpoint.  Author of voting system script.  Contributed to development of word2vec implementation, and para2vec implementation (retired and superseded by w2v+NN approach).
Kellan Childers -- Author of similarity measures and neural network (trained on word embeddings) implementation, word2vec powerpoint details. Setup github project.

SEMEVAL PROBLEM - TASK 12

Given an argument consisting of a claim and reason, select the correct warrant that explains the reasoning of the argument.  There are only two options given and only one answer is correct.  

Ex:

Topic: Are Economists Overrated?
Additional Info: Do economists have too much authority, given their mixed record at forecasting and planning?
Argument: 
Economics is not science. (Reason)
Warrant 1: non-scientific fields can’t be useful
Warrant 2: non-scientific fields can be useful
... thus, Economists are overrated (Claim)

Warrant 1 most strongly supports the argument.  Warrant 2 would suggest that economists are not necessarily overrated.

Amplifying information from the SemEval Task web site: The challenging factor is that both options are plausible and lexically very close while leading to contradicting claims. We created a new freely licensed dataset based on authentic arguments from news comments.

The two approaches taken are introduced and outlined below. The approaches were tested individually and as inputs to a voting system, arbitrating based on confidence measures.  

******************************************************************
Rule Based Implementation:
This approach utilizes sentiment values to determine which arguement to choose. A very basic example of sentiment would be associating bad or good with positive or negative. Bad -> Negative and Good -> Positive. 

The assumption is being made (after manual inspection of the data) that cohesive arguements contain the same rhetoric. 

Algorithm:
1. Execute the command perl semeval_script.pl <text file from Semeval_Data> <SentiWords_1.0 text file>(see semeval_script on how to acquire this data
    1.1 Optionally perl semeval_script.pl dev-full.txt SentiWords_1.0.txt > output.file
2. Read in dev-full.txt or train-full.txt; while being read in the data is sanitized (lowercase, \n removal, number values (0-9)), part of speech tagged using Lingua::EN::Tagger, and added to the datahash where the key is the ID.
2.1 Once all the values in the datahash for a individual entry have been sent data_set_tag_mapping is called. Subroutine data_set_tag_mapping is used for mapping the tags from Lingua::EN::Tagger into the SentiNet_1.0 tags.
3. Calls sentiment_value_tagging() subroutine.
    3.1 Loop through each value in the DataHash (these will be all the items in dev-full.txt or train-full.txt)
        3.1.1 Loop through each value of SentiWordHash and call data_value_tagging subroutine for Reason, Claim, Warrant0, and Warrant1.
        3.1.2 data_value_tagging takes in the datahashID key, the datahash attribute, and SentiWordHashID.
        3.1.3 Once these values are passed through the datahash text is parsed for a match.
            3.1.3.1 If the word is found then replace the word with the word and value. I.E. trust(.125)
4. Call data_non_tagged_words this subroutine is used to implement the backoff model, for word that have not been tagged attempt to remove -ed, -s, -ing and try to retag.
5. Call sentiment_value_calc subroutine, this subroutine calculates the sentiment value for reason, claim, and the warrants.
    5.1 If the text contains one of the key negation words, then all values after the negation are negated. I.E. not trusted(.125) -> not trusted(-.125).
    5.2 If the hash item has sentiment values then begin to calculates
        5.2.1 Call count subroutine this subroutine returns the number of words in the text which is used for normalization
6. Call evalute_answer, this subroutine goes through each element in the DataHash and compares the assigned value versus the correct answer. It also calculates the confidence of the answer which is denoted by 1 - (the picked warrant).
7. Call accuracy, this subroutine calculates the correct number of labels and prints out the accuracy.
8. Call output_confidence_csv, this subroutine prints out the ID, answer, and confidence interval to perl_confidence.txt to be used in the voting system.
    

COMMAND TO RUN PROGRAM OR INPUT:
perl semeval_script.pl .\Semeval_Data\train-full.txt .\SentiWords_1.0\SentiWords_1.0.txt

or

perl semeval_script.pl .\Semeval_Data\dev-full.txt .\SentiWords_1.0\SentiWords_1.0.txt
    
-------------- Total Labels --------------
Correct: 613
Total Labels: 1211
Overall Accuracy: 50.6193228736581

*******************************************************************
Word2Vec Implementation:
This approach attempts to look at the relatedness between the two warrants, the reason, and the claim.  The hypothesis is that the rhetoric, idiolect, and sentiment will be consistent in a cohesive argument.  

Algorithm:
1. Load the dataset, and separate it into claim (everything except warrants & tag), warrants, and true tag.
2. Find the unigrams for each warrant and claim and use these as the warrants and claims from now on.
3. Remove words common to each warrant.  
4. Replace n't with not.
5. Train a word2vec network using the GoogleNews corpus.
6. For each argument
    1) Use the word2vec model to generate word embeddings for the "pretext" (reason+claim) and each warrant.
    2) Sum together the word vectors in the claim and warrants, leaving each as a single vector.
    3) Normalize the claim and warrant vectors so they are each of magnitude 1.
7. Train a neural network composed of an input layer, two ReLU layers of 128 neurons each and a softmax binary output layer.
    1) The word embeddings (each of length N) for pretext, warrant 1, and warrant 2 are concatenated, creating an 3Nx1 "image" of the arguments to be fed to the input layer.
    2) Dataset is split into 10 folds for cross-validation.
    3) Test on each fold left out.

Executing the program:
Run python3 evaluator.py or runit.sh
Alternatively, run python3 evaluator.py -h to view command line arguments.
The program may delay for a long time at the "Loading the word embedder" step because of the size of the data.

Sample output:
--------------------------------------------------------------------
Predicting via majority decider
Baseline accuracy: 51.07 %

Loading the word embedder
Generating similarity measures between warrants and arguments
Similarity accuracy: 59.59 %

Predicting via word embedder and neural network using 10 fold cross validation
Using TensorFlow backend.
/usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
2017-12-09 01:08:26.793824: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX
1089/1089 [==============================] - 0s 56us/step
1089/1089 [==============================] - 0s 65us/step
1089/1089 [==============================] - 0s 78us/step
1089/1089 [==============================] - 0s 74us/step
1089/1089 [==============================] - 0s 91us/step
1089/1089 [==============================] - 0s 95us/step
1089/1089 [==============================] - 0s 109us/step
1089/1089 [==============================] - 0s 112us/step
1089/1089 [==============================] - 0s 114us/step
1089/1089 [==============================] - 0s 133us/step
Neural Network Accuracy: 57.44%

