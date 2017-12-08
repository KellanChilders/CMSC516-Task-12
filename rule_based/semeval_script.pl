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
#     SentiWords_1.0 - Guerini M., Gatti L. & Turchi M. “Sentiment Analysis: How to Derive Prior Polarities from SentiWordNet”. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP'13), pp 1259-1269. Seattle, Washington, USA. 2013.
#
########################################################################
#
#	ROLES
#	Megan Davis - Author of semeval_script.pl, and rules based implementation within powerpoint. Set up trello board.
# 
#
#  ALL CODE IN THIS PROGRAM IS AUTHORED BY MEGAN DAVIS UNLESS OTHERWISE SPECIFIED
#####################################################################
#	
#	How to Run:
#   1) Have semeval_script.pl, SentiWords_1.0 (can be found here https://hlt-nlp.fbk.eu/technologies/sentiwords), and dev-full.txt
#	2) Run 
#			perl semeval_script.pm <dev or train txt> <lexicon (sentiwords)>
#			perl semeval_script.pl .\Semeval_Data\dev-full.txt .\SentiWords_1.0\SentiWords_1.0.txt
#
#######################################################################
#
#	ALGORTHIM:
#	The algorithm follows the directions of those in the README file. Additional comments have been provided in the code. But ultimately reference the README points.
#
###############################################################################

### USE ###
use warnings;
use strict;
use Data::Dumper qw(Dumper);
use Lingua::EN::Tagger;

$Data::Dumper::Sortkeys = 1;

### Files: Currently utilizes two files dev-full.txt or train-full.txt, and SentiNet_1.0 ###
my $filename = $ARGV[0];
my $filename2 = $ARGV[1];

my $p = new Lingua::EN::Tagger; ### For a list of the tags it used (Penn Treebank) reference http://cpansearch.perl.org/src/ACOBURN/Lingua-EN-Tagger-0.28/README ###
my %DataHash; ### Hash containing all the data from the dev-full.txt or train-full.txt ###
my %SentiWordHash; ### Hash containing all the data from SenitNet_1.0 ###

### Open the given file, in this case dev-full.txt or train-full.txt and preform part of speech tagging and tag mapping ###
open(my $fh, '<', $filename) or die "Could not open";	
while(my $row = <$fh>)
{
	my @temparray = split('\t',$row);

	### Claim ###
	$DataHash{$temparray[0]}{Claim} = text_sanitation($temparray[4]);
	$DataHash{$temparray[0]}{Claim_Tagged} = $p->get_readable(text_sanitation($temparray[4]));

	### Reason ###
	$DataHash{$temparray[0]}{Reason} = text_sanitation($temparray[5]);
	$DataHash{$temparray[0]}{Reason_Tagged} = $p->get_readable(text_sanitation($temparray[5]));

	### Warrant0 ###
	$DataHash{$temparray[0]}{Warrant0} = text_sanitation($temparray[1]);
	$DataHash{$temparray[0]}{Warrant0_Tagged} = $p->get_readable(text_sanitation($temparray[1]));
	
	### Warrant1 ###
	$DataHash{$temparray[0]}{Warrant1} = text_sanitation($temparray[2]);
	$DataHash{$temparray[0]}{Warrant1_Tagged} = $p->get_readable(text_sanitation($temparray[2]));
	
	### Value Assignment ###
	$DataHash{$temparray[0]}{Warrant0_Value} = 0;
	$DataHash{$temparray[0]}{Warrant1_Value} = 0;
	$DataHash{$temparray[0]}{Reason_Value} = 0;
	$DataHash{$temparray[0]}{Claim_Value} = 0;
	$DataHash{$temparray[0]}{CorrectLabel} = $temparray[3];
	$DataHash{$temparray[0]}{Debate_Title} = $temparray[6];
	$DataHash{$temparray[0]}{Debate_Info} = $temparray[7];
	$DataHash{$temparray[0]}{Answer} = -1;

	### Call to convert Tagger tags to SentiNet tags ###
	data_set_tag_mapping($temparray[0], 'Reason_Tagged');
	data_set_tag_mapping($temparray[0], 'Claim_Tagged');
	data_set_tag_mapping($temparray[0], 'Warrant0_Tagged');
	data_set_tag_mapping($temparray[0], 'Warrant1_Tagged');
}
 
### Opening and loading in the sentiment data from SentiNet_1.0 ###
open($fh, '<', $filename2) or die "Could not open SentiWordNet File.";	
while(my $row = <$fh>)
{
	if($row =~ m/(.*)\s-?+[0-9.]{2,}\n$/) ### Only add words that a sentiment value greater or less than 0 ###
	{
		$row =~ s/\n//g;
		my @temparray = split('\t',$row);
		$SentiWordHash{$temparray[0]} = $temparray[1];
	}
}

### Subroutines Called, check individual subroutines to see additional calls ###
sentiment_value_tagging();
evaluate_answer(); 
accuracy(); 
output_confidence_csv();
#print_hash(); #-- Can be used to print out the hases

############# Sub Routines ######################
#
# THESE ARE ORDERED IN ORDER OF CALLS IN THE PROGRAM
#
#################################################

### Sanatizes the text passed in passed in. ###
sub text_sanitation
{
	my $my_text = $_[0];
	$my_text =~ s/[0-9]{1,}[A-Za-z]+//g;
	#$my_text =~ s/[[:punct:]]//g;
	$my_text =~ s/-/ /g;
	$my_text =~ s/\n+/\n/g;
	$my_text =~ s/\s+/ /g;
	$my_text = lc($my_text);
	return $my_text;
}

### Used to map the tags from Perl module Tagger to SentiNet's tags ###
sub data_set_tag_mapping
{
	my ($id, $key2) = @_;
	
	$DataHash{$id}{$key2} =~ s/\/(NNP|NNPS|NNS|NN)/#n/g;
	$DataHash{$id}{$key2} =~ s/\/(RBR|RBS|RP|RB)/#r/g;
	$DataHash{$id}{$key2} =~ s/\/(CD|JJR|JJS|JJ)/#a/g;
	$DataHash{$id}{$key2} =~ s/\/(MD|VBN|VBD|VBG|VBP|VBZ|VB)/#v/g;	
}

### Calls data_value_tagging to add the sentiment values to the data from SentiNet, and then calculates the values from ###
sub sentiment_value_tagging
{
	foreach my $k (keys %DataHash)
	{
		foreach my $k2 (keys %SentiWordHash) 
		{
			data_value_tagging($k, 'Reason_Tagged', $k2);
			data_value_tagging($k, 'Claim_Tagged', $k2);
			data_value_tagging($k, 'Warrant0_Tagged', $k2);
			data_value_tagging($k, 'Warrant1_Tagged', $k2);
		}

	data_non_tagged_words($k, 'Reason_Tagged');
	data_non_tagged_words($k, 'Claim_Tagged');
	data_non_tagged_words($k, 'Warrant0_Tagged');
	data_non_tagged_words($k, 'Warrant1_Tagged');

	sentiment_value_calc($k,'Reason_Tagged', 'Reason_Value');
	sentiment_value_calc($k,'Claim_Tagged', 'Claim_Value');
	sentiment_value_calc($k,'Warrant0_Tagged', 'Warrant0_Value');
	sentiment_value_calc($k,'Warrant1_Tagged', 'Warrant1_Value');
	}
}

### Assign the sentiment value from SentiNet_1.0 to the matching word ###
sub data_value_tagging
{
	my ($key1, $key2, $sentikey) = @_;

	if($DataHash{$key1}{$key2} =~ m/\b($sentikey)\b/)
	{
		my $v = $SentiWordHash{$sentikey};
		$DataHash{$key1}{$key2} =~ s/\b($sentikey)\b/$1\($v\)/g; ### Note for improvement, may be faster to do if exists(?)
	}
}

### Back off model for -ed, -ing, -s that is used to tag words that do not have values, i.e trusted -> trust ###
sub data_non_tagged_words
{
	my ($key1, $key2) = @_;
	my @temp = split(' ', $DataHash{$key1}{$key2});

	for (my $array_element = 0; $array_element < scalar @temp; $array_element++)
	{
		if($temp[$array_element] =~ m/#/)
		{
			my @hash_split = split('#', $temp[$array_element]);

			### If the word has a value (which we know based on the fact of if the word contains a (, which would be there from a the value (.25)) ###
			if($hash_split[1] =~ m/\(/)
			{}
			else
			{				
				#print "================\n";
				#print "Before:" . $hash_split[0] ."\n";
				$hash_split[0] =~ s/([a-z]*)(ed)$/$1/g;
				$hash_split[0] =~ s/([a-z]*)(ing)$/$1/g;
				$hash_split[0] =~ s/([a-z]*)(s)$/$1/g;
				#print "After:" . $hash_split[0] ."\n";
				#print "================\n";
			}

			$temp[$array_element] = join ('#', @hash_split);
			if(exists $SentiWordHash{$temp[$array_element]})
			{
				$temp[$array_element] .= "(".$SentiWordHash{$temp[$array_element]}.")";
			}
		}
	}
	$DataHash{$key1}{$key2} = join(' ', @temp);
}

### Calculates the value of the string passed into it for Reason, Warrant0, and Warrant1. ###
### The negation process works as follows, if a negation term is found in the sentence it will negate all the words follow the negation term. ###
sub sentiment_value_calc
{
	my ($key1, $key2, $value) = @_;

	### Negation ###
	if($DataHash{$key1}{$key2} =~ /\b(n't#r|not#r|cannot#n|cannot#v|not#n|no\/DET)\b/)
	{
		#print "------\n";
		#print "Before: \n";
		my @split_text = split(' ',$DataHash{$key1}{$key2});
		#print Dumper \@split_text;
		for (my $array_element = 0; $array_element < scalar @split_text; $array_element++)
		{
			if($split_text[$array_element] =~ /\b(n't#r|not#r|cannot#n|cannot#v|not#n|no\/DET)\b/)
			{
				for (my $sub_loop = $array_element + 1; $sub_loop < scalar @split_text; $sub_loop++)
				{
					if($split_text[$sub_loop] =~ /\(-/)
					{
						$split_text[$sub_loop] =~ s/\(-/\(/g;
					}
					elsif($split_text[$sub_loop] =~ /\(/)
					{
						$split_text[$sub_loop] =~ s/\(/\(-/g;
					}			
				}
			}
		}
		#print "\nAfter:\n ";
		#print Dumper \@split_text;
		$DataHash{$key1}{$key2} = join(" ", @split_text);
	}

	### Calculation ###
	my @matches = ($DataHash{$key1}{$key2} =~ /-?[0-9]+\.?[0-9]+/g);
	if((scalar @matches) > 0)
	{
		foreach my $num (@matches)
		{	
			$DataHash{$key1}{$value} += $num;
		}
		$DataHash{$key1}{$value} = $DataHash{$key1}{$value} / count($DataHash{$key1}{$key2}); ###Normalizing###
	}
}

### Counts the amount of words in a sentence. This is used for normalization. ###
sub count
{
    my $text = $_[0]; 
    my $count = 0;
    $count++ while $text =~ /\S+/g; 
    return $count;
}

sub evaluate_answer
{
	#my $equal = 0;	
	foreach my $k(keys %DataHash)
	{
		#my $warrant_0 = abs(($DataHash{$k}{Reason_Value} + $DataHash{$k}{Claim_Value}) - $DataHash{$k}{Warrant0_Value});
		#my $warrant_1 = abs(($DataHash{$k}{Reason_Value} + $DataHash{$k}{Claim_Value}) - $DataHash{$k}{Warrant1_Value});

		### Only using the reason because it was shown to preform better ###
		my $warrant_0 = abs($DataHash{$k}{Reason_Value} - $DataHash{$k}{Warrant0_Value});
		my $warrant_1 = abs($DataHash{$k}{Reason_Value} - $DataHash{$k}{Warrant1_Value});

		if($warrant_0 < $warrant_1)
		{
			$DataHash{$k}{Answer} = '0';
			$DataHash{$k}{Confidence} = 1 - $warrant_0;
		}	
		elsif ($warrant_1 < $warrant_0)
		{
			$DataHash{$k}{Answer} = '1';
			$DataHash{$k}{Confidence} = 1 - $warrant_1;
		}
		else
		{
			$DataHash{$k}{Answer} = '1';	
			#$equal++;
			$DataHash{$k}{Confidence} = 0;

		}
    }
	#print "Instances in which warrants = 0.... : $equal \n";
}


### Checks assinged value versus the correct lable. Prints out accuracy and final information. ###
sub accuracy{
	my $totalLabels = 0;
	my $correctTotalLabels = 0;
	foreach my $key (keys %DataHash)
	{
		if($DataHash{$key}{Answer} eq $DataHash{$key}{CorrectLabel} )
		{
			$correctTotalLabels++;
		}
=begin
		else
		{

			print "------------------------\n";
			print "ID: $DataHash{$key}{ID} \n ";
			print "Debate Title: $DataHash{$key}{Debate_Title}\n";
			print "Debate Info: $DataHash{$key}{Debate_Info} ";
			print "Claim : $DataHash{$key}{Claim_Tagged}\n ";
			print "Claim : $DataHash{$key}{Claim_Value}\n ";
			print "Reason: $DataHash{$key}{Reason_Tagged}\n";
			print "Reason: $DataHash{$key}{Reason_Value}\n";
			print "Warrant 0: $DataHash{$key}{Warrant0_Tagged}\n ";
			print "Warrant0 Value: $DataHash{$key}{Warrant0_Value}\n";									
			print "Warrant 1: $DataHash{$key}{Warrant1_Tagged}\n";
			print "Warrant1 Value: $DataHash{$key}{Warrant1_Value}\n";
			print "CorrectLabel: $DataHash{$key}{CorrectLabel}\n";	
			print "Answer: $DataHash{$key}{Answer}\n";
			print "------------------------\n";
		}
=end
=cut
		$totalLabels++;
	}

	my $accuracy = ($correctTotalLabels / $totalLabels) * 100;

	print "-------------- Total Labels --------------\n";
	print "Correct: $correctTotalLabels \n";
	print "Total Labels: $totalLabels \n";
	print "Overall Accuracy: $accuracy";
}




### Opens a file to print out the ID, TAG, CONFIDENCE; this is to be used in a voting system of other methods attempted ###
sub output_confidence_csv
{
	# ID, TAG, CONFIDENCE
	my $csv_file = 'perl_confidence.txt';
	open(my $fh, '>', $csv_file) or die "Could not open file '$csv_file' $!";
	foreach my $key (keys %DataHash)
	{
		my $string = $key .",".$DataHash{$key}{Answer}.",".$DataHash{$key}{Confidence}."\n";
		print $fh "$string";
	}
	close $fh;
}

### Prints out hash using dumper ###
sub print_hash
{
	print "########################################\n";
	#print Dumper \%DataHash;
	#print Dumper \%SentiWordHash;
}