#!/usr/bin/perl -w
######################################################################
#
#	semeval_script.pl
#	Megan Davis
#	10/7/2017
#
#   Utilizing Code from: 
#  	   Lingua::EN::Tagger
#   Text Used:   
#     SCL-NMA -- http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm ### Semeval-2016 Task 7: Determining Sentiment Intensity of English and Arabic Phrases. Svetlana Kiritchenko, Saif M. Mohammad, and Mohammad Salameh. In Proceedings of the International Workshop on Semantic Evaluation (SemEval ’16). June 2016. San Diego, California.
#
########################################################################
#
#	ROLES
#	Megan Davis - Author of semeval_script.pl, and procedural implementation within powerpoint. Set up trello board.
#   Andrew Ward - Author of task information/introduction and proposed approach. In addition to providing suggestions on solutions implemented. (Also implemented code not included in code drop #1)
#	Kellan Childers -- Author of word2vec implementation, word2vec powerpoint details. Setup github project.
#
#  ALL CODE IN THIS PROGRAM IS AUTHORED BY MEGAN DAVIS UNLESS OTHERWISE SPECIFIED
#####################################################################
#	
#	How to Run:
#   1) Have semeval_script.pl, SCL-NMA.txt, and dev-full.txt
#	2) Run perl semeval_script.pl dev-full.txt SCL-NMA/SCL-NMA.txt
#
#	ONLY TESTED ON WINDOWS 10  
#
#######################################################################
#
#
#	ALGORTHIM:
#	The algorithm follows the directions of those in the README file. Additional comments have been provided in the code. But ultimately reference the README points.
#
###############################################################################



# USE 
use warnings;
use strict;
use Data::Dumper qw(Dumper);
use Lingua::EN::Tagger; ## Currently not used in code #1 implementation

$Data::Dumper::Sortkeys = 1;

# Files: Currently utilizes two files dev-full.txt, SCL-NMA.txt
my $filename = $ARGV[0];
my $filename2 = $ARGV[1];

# Variables
my %OverallHash; #Hash containing all the data
my %SentimentHash; #Hash containing the data from SCL-NMA.txt
my %Results;
my $num = 0; #Used assign a unique ID to hashes
my $negated_changed_values = 0;


# Variables Not Being Used -- but still declaring to avoid run errors.
my $p = new Lingua::EN::Tagger;
my %unigram;
my $unigram_frequency = 0;
my $all_text;
my %bigram;
my $bigram_frequency = 0;

################ 2 ####################################
#Open Dev-Full Text and read in the data/ preform text manipulations 
open(my $fh, '<', $filename) or die "Could not open";	
while(my $row = <$fh>)
{
	my @temparray = split('\t',$row);

	# Not all data in this hash is being used but created for later implementation. 
	$OverallHash{$num}{ID} = $temparray[0];
	$OverallHash{$num}{Warrant0} = text_sanitation($temparray[1]);#$p->get_readable($p->add_tags($temparray[1]));
	$OverallHash{$num}{Warrant0_Sentiment} = 0;
	$OverallHash{$num}{Warrant1_Sentiment} = 0;
	$OverallHash{$num}{Warrant1} = text_sanitation($temparray[2]);#$p->get_readable($p->add_tags($temparray[2]));
	$OverallHash{$num}{CorrectLabel} = $temparray[3];
	#$OverallHash{$num}{Reason_tagged} = $p->get_readable($p->add_tags($temparray[4]));
	$OverallHash{$num}{Reason_Claim_Combined} = text_sanitation($temparray[4] . " , " .+ $temparray[5]);
	$OverallHash{$num}{Claim} = text_sanitation($temparray[4]);
	$OverallHash{$num}{Reason} = text_sanitation($temparray[5]);
	#$OverallHash{$num}{Claim_tagged} = $p->get_readable($p->add_tags($temparray[5]));
	$OverallHash{$num}{Debate_Title} = $temparray[6];
	$OverallHash{$num}{Debate_Info} = $temparray[7];
	$OverallHash{$num}{Answer} = -1;
	$OverallHash{$num}{sentiment_value} = 0;

	$num++;

	### NOT BEING USED ###########
	$all_text .= " " .$temparray[1];
	$all_text .= " " .$temparray[2];
	$all_text .= " " .$temparray[4];
	$all_text .= " " .$temparray[5];
	$all_text .= " " .$temparray[6];
	$all_text .= " " .$temparray[7];
	###############################
	
}

#Reset Num
$num = 0;

## Open SCL-NMA Text file -- this has SO values for about 3200 items
open($fh, '<', $filename2) or die "Could not open";	
while(my $row = <$fh>)
{
	my @temparray = split('\t',$row);
	$SentimentHash{$num}{word} = $temparray[0];
	$SentimentHash{$num}{score} = $temparray[1];

	$num++;
}
########################################################

#Subroutines called#
#Algorithm steps steps 3, 4, 5, 6
reason_claim_sentiment_value(); #3
warrant_sentiment_set(); #4
COMAPRISION_SHOWDOWN(); #5
accuracy(); #6

#print_hash(); #-- Can be used to print out OverallHash


############# Sub Routines ######################
#
#		There are two sub routine sections -- USED and NOT USED.
#		THESE ARE ORDER IN ALPHABETICAL ORDER.  
#		
#		USED -- Are currently active within the current runnable program
#		NOT USE -- Subroutines created that are not used in the current runnable program (But not worth deleting in case brought back to be used)
#
#
#################################################

######################## USED ###########################

#Calculating Sentiment Value
# Algorithm Step #3
# 1) For each of the sentiment values for the Reason_Claim_Combined
# 2) Adds values together to determine if negative or positive overall
sub reason_claim_sentiment_value{
	#print "Beginning Sentiment Values\n";
	foreach my $k (keys %OverallHash)
	{
		foreach my $k2 (keys %SentimentHash) 
		{
			my $w = $SentimentHash{$k2}{word}; #assign word from sentiment to variable

			if($OverallHash{$k}{Reason_Claim_Combined} =~ /$w/) #if the sentiment contains the word/phrase add the sentiment value
			{

				$OverallHash{$k}{sentiment_value} += $SentimentHash{$k2}{score};
			}
		}

		my @matches = $OverallHash{$k}{Reason_Claim_Combined} =~ /(not|isnt|shouldnt|wouldnt|cant|while|yet)/g;
		#print scalar @matches;
		#print "\n";
		if((scalar @matches) > 0)
		{
			$OverallHash{$k}{sentiment_value} = $OverallHash{$k}{sentiment_value} * (-1 ** (scalar @matches));	
		}
	}	
}

#Calculate Warrant Sentiment
# Algorithm Step #4
# 1) For each of the warrants within a claim calculate the sentiment values
# 2) Adds values together to determine negative or positive overall for the warrant
sub warrant_sentiment_set{
	#print "Beginning Sentiment Warrants\n";
	foreach my $x (keys %OverallHash)
	{
		foreach my $z (keys %SentimentHash) 
		{
			my $lt = $SentimentHash{$z}{word}; #assign word/phrase from sentiment to variable
			#print "$t : $w\n";
			#Algorithm step 4.1.1
			if($OverallHash{$x}{Warrant0} =~ /$lt/)
			{
				$OverallHash{$x}{Warrant0_Sentiment} += $SentimentHash{$z}{score};
			}
			#Algorithm step 4.1.1
			if($OverallHash{$x}{Warrant1} =~ /$lt/)
			{
				$OverallHash{$x}{Warrant1_Sentiment} += $SentimentHash{$z}{score};	
			}
		}
	#FIX THIS	
	#Algorithm Step 4.2 -- Negation of value if it contains negating words. Determined the number of negation words and negate based off that/
	my @matches = $OverallHash{$x}{Warrant0} =~ /\b(cannot|not|isnt|shouldnt|wouldnt|cant|yet|wont|havent)\b/g;
	#print scalar @matches;
	#print "\n";
	if((scalar @matches) > 0)
	{
		$OverallHash{$x}{Warrant0_Sentiment} = $OverallHash{$x}{Warrant0_Sentiment} * (-1 ** (scalar @matches));
	}
	
	@matches = $OverallHash{$x}{Warrant1} =~ /\b(cannot|not|isnt|shouldnt|wouldnt|cant|yet|wont|havent)\b/g;
	if((scalar @matches) > 0)
	{
		$OverallHash{$x}{Warrant1_Sentiment} = $OverallHash{$x}{Warrant1_Sentiment} * (-1 ** (scalar @matches));
	}
	
	}	
}

#COMAPRISION SHOW DOWN (CAPS BECAUSE THIS IS IMPORTANT)
# Algorithm Step #5
# 1) Calculate which warrant is closer to the claim_reason_sentiment
# 2) Assigns the answer to Hash for comparision
sub COMAPRISION_SHOWDOWN{
	my $equal = 0;	
	foreach my $yellow(keys %OverallHash)
	{
		#Algorithm Step #5.1
		my $warrant_0 = abs($OverallHash{$yellow}{sentiment_value} - $OverallHash{$yellow}{Warrant0_Sentiment});
		my $warrant_1 = abs($OverallHash{$yellow}{sentiment_value} - $OverallHash{$yellow}{Warrant1_Sentiment});

		#print "$Z1 : $Z2\n";

		#Algorithm Step #5.2.1
		if($OverallHash{$yellow}{sentiment_value} == 0)
		{
			$equal++;
		}
		if($warrant_0 < $warrant_1)
		{
			$OverallHash{$yellow}{Answer} = '0';
		}	
		elsif ($warrant_1 < $warrant_0) #Algorithm Step #5.2.2
		{
			$OverallHash{$yellow}{Answer} = '1';
		}
		else #Algorithm Step #5.2.3
		{
			$OverallHash{$yellow}{Answer} = '1';	
			$equal++;
		}
	}
	print "Instances in which the Warrants are of equal sentiment value or Claim+Reason sentiment value = 0: $equal \n";
}

#Checks how many claims were accurately tagged (currently using randomness baseline) 
#Algorithm Step #6
#-------------- Total Labels --------------
#Correct: 157 
#Total Labels: 317 
#Overall Accuracy: 49.5268138801262 
#The above data was generated using random_sample
sub accuracy{
	my $totalLabels = 0;
	my $correctTotalLabels = 0;
	foreach my $key (keys %OverallHash)
	{
		if($OverallHash{$key}{Answer} eq $OverallHash{$key}{CorrectLabel} )
		{
			$correctTotalLabels++;
		}
		else
		{
			#print "------------------------\n";
			#print "ID: $OverallHash{$key}{ID} \n ";
			#print "Debate Title: $OverallHash{$key}{Debate_Title}\n";
			#print "Debate Info: $OverallHash{$key}{Debate_Info} ";
			#print "Claim : $OverallHash{$key}{Claim}\n ";
			#print "Reason: $OverallHash{$key}{Reason}\n ";
			#print "Claim + Reason: $OverallHash{$key}{Reason_Claim_Combined}\n ";
			#print "Warrant 0: $OverallHash{$key}{Warrant0}\n ";
			#print "Warrant 1: $OverallHash{$key}{Warrant1}\n";
			#print "CorrectLabel: $OverallHash{$key}{CorrectLabel}\n";	
			#print "Answer: $OverallHash{$key}{Answer}\n";
			#print "Claim + Reason Sentiment: $OverallHash{$key}{sentiment_value}\n";
			#print "Warrant0 Value: $OverallHash{$key}{Warrant0_Sentiment}\n";					
			#print "Warrant1 Value: $OverallHash{$key}{Warrant1_Sentiment}\n";
			#print "------------------------\n";
		}
		$totalLabels++;
	}


	my $accuracy = ($correctTotalLabels / $totalLabels) * 100;

	print "-------------- Total Labels --------------\n";
	print "Correct: $correctTotalLabels \n";
	print "Total Labels: $totalLabels \n";
	print "Overall Accuracy: $accuracy";
}

#Prints out hash using Dumper
sub print_hash{
	#my %tempHash = $_[0];
	print "########################################\n";
	#print Dumper \%unigram;
	print "########################################\n";
	#print Dumper \%bigram;
	print "########################################\n";
	print Dumper \%OverallHash;
	print "########################################\n";
	#print Dumper \%SentimentHash;
}

## Sanatizes Text
sub text_sanitation{
	my $my_text = $_[0];
	$my_text =~ s/[[:punct:]]//g;
	$my_text =~ s/\n+/\n/g;
	##############################################################################################
	######## Stop Words remove too much information for positive negative analysis ###############
	#my $t = stop_word_removal();
	#$my_text =~ s/$t//g;
	##############################################################################################
	$my_text =~ s/\s+/ /g;
	$my_text = lc($my_text);
	return $my_text;
}



################################################
#		NOT USED SUB ROUTINES
###############################################

#attempted negation method
sub negate{
	
	my $id = shift;


	my $warrant_0 = abs($OverallHash{$id}{sentiment_value} - $OverallHash{$id}{Warrant0_Sentiment});
	my $warrant_1 = abs($OverallHash{$id}{sentiment_value} - $OverallHash{$id}{Warrant1_Sentiment});

	if($warrant_0 < $warrant_1)
		{

			$OverallHash{$id}{Answer} = '0';
			$negated_changed_values++;
		}	
		elsif ($warrant_1 < $warrant_0) #Algorithm Step #5.2.2
		{
			$OverallHash{$id}{Answer} = '1';
			$negated_changed_values++;
		}
		else #Algorithm Step #5.2.3
		{
			$OverallHash{$id}{Answer} = '-1';	
		}	

}

# creates a bi-gram of $all_text
sub create_bigram{
	my @words = split(/\s/, $all_text);
	for(my $i = 0; $i < $#words; $i++)
	{
		my $temp_text = $words[$i] . " " . $words[$i + 1];	
		if(!exists($bigram{$temp_text}))
		{
			$bigram{$temp_text} = 1;
		}
		else
		{
			$bigram{$temp_text}++;
		}
		$bigram_frequency++;		
	}
}

#Creates a Unigram out of $all_text
sub create_unigram{
	my @words = split(/\s/, $all_text);
	for(my $i = 0; $i <= $#words; $i++)
	{
		if(!exists($unigram{$words[$i]}))
		{
			$unigram{$words[$i]} = 1;
		}
		else
		{
			$unigram{$words[$i]}++;
		}
		$unigram_frequency++;
	}
}

### THis bigram only pulls text that has certain tags -- Tag selection pulled from Thumbs Up Thumbs Down Paper
sub create_tagged_bigram{
	my @words = split(/\s/, $all_text);
	for(my $i = 0; $i < $#words; $i++)
	{
		my $temp_text = $words[$i] . " " . $words[$i + 1];	
		if($words[$i] =~ /(NN|NNS)/)
		{
			if($words[$i+1] =~ /JJ/)
			{
				if(!exists($bigram{$temp_text}))
				{
					$bigram{$temp_text} = 1;
				}
				else
				{
					$bigram{$temp_text}++;
				}
				$bigram_frequency++;
			}
		}
		elsif($words[$i] =~ /JJ/)
		{
			if($words[$i+1] =~ /(JJ|NNS|NN)/)
			{
				if(!exists($bigram{$temp_text}))
				{
					$bigram{$temp_text} = 1;
				}
				else
				{
					$bigram{$temp_text}++;
				}
				$bigram_frequency++;
			}
		}
	}
}

#Calculates PMI 
#Must create unigram first
#Pass in Word1 and Word2
sub pointwise_mutual_information{
	#info from unigram
	my $word1= shift;
	my $word2 = shift;
	#info from bigram
	my $co_occur = shift;
	
	$word1 = $word1 / $unigram_frequency;
	$word2 = $word2 / $unigram_frequency;

	$co_occur = $co_occur / $bigram_frequency;

	print "Word1 Prob: $word1 \n";
	print "Word2 Prob: $word2 \n";
	print "Co_Occur Prob: $co_occur \n";

	my $PMI = log($co_occur / ($word1 * $word2))/log(2);

	print "PMI: $PMI";
}

#Prints out Overall Hash -- missing some of the values, hasn't been updated
sub print_out_overall_hash{
	foreach my $key(keys %OverallHash)
	{
		print "------------------------";
		print "$OverallHash{$key}{ID} ~ ";
		print "$OverallHash{$key}{Debate_Title} ~ ";
		print "$OverallHash{$key}{Debate_Info} ~ ";
		print "$OverallHash{$key}{Warrant0} ~ ";
		print "$OverallHash{$key}{Warrant1} ~";
		print "$OverallHash{$key}{CorrectLabel} ~";
	}
}

#Uses rand to 'guess' the answers
sub random_sample{
	foreach my $key (keys %OverallHash)
	{
		my $tempRandom = rand();
		if($tempRandom > .5)
		{
			$OverallHash{$key}{Answer} = '1';
		}
		else
		{
			$OverallHash{$key}{Answer} = '0';
		}		
	}
}


#Calculate Semantic Orientation -- Empty Subroutine  
sub semantic_orientation{
	my $pmi_one = shift;
	my $pmi_two = shift;
}

