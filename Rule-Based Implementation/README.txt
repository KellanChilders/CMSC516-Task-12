README DOCUMENTATION:

AUTHORS: Megan Davis, Kellan Childers, Andrew Ward

ROLES

Megan Davis - Author of semeval_script.pl, and procedural implementation within powerpoint. Set up trello board.
Andrew Ward - Author of task information/introduction and proposed approach. In addition to providing suggestions on solutions implemented. (Also implemented code not included in code drop #1)
Kellan Childers -- Author of word2vec implementation, word2vec powerpoint details. Setup github project.

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

READ THIS NOTE: Due to the way the data is defined Warrant0 EQUALS warrant0, warrant0 EQUALS warrant1; this applies when being talked about in the algorithm and the code. This change can be confusing. There are plans to normalize this convention code drop 2.

Algorthim: 
1. Execute the command perl semeval_script.pl dev-full.txt SCL-NMA/SCL-NMA.txt -- dev-full.txt contains the problem data provided by semeval, SCL-NMA.txt is a sentiment lexicon by Saif M. Mohammad it ranks phrases as positive or negative.
	1.1 Optionally perl semeval_script.pl dev-full.txt SCL-NMA.txt > output.file
2. The two text files passed in will be parsed for data, including Warrants, Claim, Reason, Answer, and Reason_Claim combined.
3. Call reason_claim_sentiment_value() -- This subroutine with take the reason and claim (which have been combined) and evaluate for the sentiment.
	3.1) Each item with in the data is hit. 
	3.2) The reason_claim_combined is checked for a match within all the phrases/words within SCL-NMA.
		3.2.1) If a match is found the sentiment_value is updated by adding the matched phrase/word to sentiment value.
		3.2.2) If not, nothing happens and enters the next iterration..
4. Call warrant_sentiment_set() -- This subroutine takes the two warrants and evaluates for the sentiment. 
	4.1) Each item with in the data is hit.
		4.1.1) If a match is found the warrantx_sentiment is updated by adding the matched phrase/word to the value.
		4.1.2) If not, nothing happens and enters the next iterration. 
	4.2) If the warrant contains not/isnt the value is negated by multiply negative one to change it from positive or negative depending on the value.
5. Call COMPARISON_SHOWDOWN() -- This subroutine evalutes the warrants to determine which is closer to the sentiment_value.
	5.1) To calculate which is closer the formula used is absolute_value(sentiment_value - warrantx_sentiment)
	5.2) If 
		5.2.1) warrant0 < warrant1 
			5.2.1.1) Select warrant0 as the answer
		5.2.2) warrant1 < warrant0
			5.2.2.1) Select warrant1 as the answer
		5.2.3) Everything else (i.e. warrant0 equals warrant1)
			5.2.3.1) Select warrant1 
6) Call Accuracy() -- This subroutine compares to the answer gives versus the correct answer. 	
	6.1) Returns the accuracy rate.
	


COMMAND TO RUN PROGRAM OR INPUT:
perl semeval_script.pl dev-full.txt SCL-NMA/SCL-NMA.txt
	
Instances in which the Warrants are of equal sentiment value or Claim+Reason sentiment value = 0: 109
-------------- Total Labels --------------
Correct: 169
Total Labels: 316
Overall Accuracy: 53.4810126582278


*******************************************************************
Word2Vec Implementation:

<introduction>

<algorithm>

<how to run>