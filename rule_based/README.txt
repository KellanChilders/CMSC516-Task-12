README DOCUMENTATION:

AUTHORS: Megan Davis

ROLES

Megan Davis - Author of semeval_script.pl, and knowledge based implementation within powerpoint. Set up trello board.

SEMEVAL PROBLEM - TASK 12

Given an argument consisting of a claim and reason, select the correct warrant that explains the reasoning of the argument.

Ex:

Topic: Are Economists Overrated?
Additional Info: Do economists have too much authority, given their mixed record at forecasting and planning?
Argument:
Economics is not science. (Reason)
Warrant 1: non-scientific fields can’t be useful
Warrant 2: non-scientific fields can be useful
... thus, Economists are overrated (Claim)


This approach utilizes sentiment values to determine which argument to choose. A very basic example of sentiment would be associating bad or good with positive or negative. Bad -> Negative and Good -> Positive.

The assumption is being made (after manual inspection of the data) that cohesive arguments contain the same rhetoric.

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