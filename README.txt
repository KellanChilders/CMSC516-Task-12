README DOCUMENTATION:

AUTHORS: Kellan Childers, Megan Davis, Andrew Ward

ROLES

Megan Davis - Author of semeval_script.pl, and procedural implementation within powerpoint. Set up trello board.
Andrew Ward - Author of task information/introduction and proposed approach. In addition to providing suggestions on solutions implemented. (Also implemented code not included in code drop #1)
Kellan Childers -- Author of similarity measures and neural network implementations, word2vec powerpoint details. Setup github project.

SEMEVAL PROBLEM - TASK 12

Given an arguement consisting of a claim and reason, select the correct warrant that explains the reasoning of the arguement.

Ex:

Topic: Are Economists Overrated?
Additional Info: Do economists have too much authority, given their mixed record at forecasting and planning?
Argument: 
Economics is not science. (Reason)
Warrant 1: non-scientific fields can’t be useful
Warrant 2: non-scientific fields can be useful
... thus, Economists are overrated (Claim)


The two approaches taken are introduced and outlined below.

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
This approach attempts to look at the relatedness between the two warrants and the claim. The hypothesis is that a warrant following the emotion of the claim would be more likely to be authored by the same individual.

Algorithm:
1. Load the dataset, and separate it into claim (everything except warrants & tag), warrants, and true tag.
2. Find the unigrams for each warrant and claim and use these as the warrants and claims from now on.
2. Remove shared words from each claim (if a word.
3. Replace n't with not.
4. Train a word2vec network using the GoogleNews corpus.
5. For each argument
	1) Use the word2vec model to embed each word in the warrants and claim.
	2) Sum together the word vectors in the claim and warrants, leaving each as a single vector.
	3) Normalize the claim and warrant vectors so they are each of magnitude 1.
	4) Take the cosine similarity between the claim and each warrant.
	6) Predict the warrant with the highest similarity, or warrant 1 if equal.

Executing the program:
Run python3 evaluator.py or runit.sh
Alternatively, run python3 evaluator.py -h to view command line arguments.