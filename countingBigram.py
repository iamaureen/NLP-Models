#reference: http://www.katrinerk.com/courses/python-worksheets/language-models-in-python

import nltk

from nltk.corpus import brown

# an nltk.FreqDist() is like a dictionary,
# but it is ordered by frequency.
# Also, nltk automatically fills the dictionary
# with counts when given a list of words.

#contains 'key' - for words, 'values' - for frequency count of words.
freq_brown = nltk.FreqDist(brown.words())
#print(freq_brown)

# #By default a FreqDist is not sorted.
# print(list(freq_brown.keys())[:20])
#
# #if we sort it and print it, it will give the top words but without the frequencies
# fdist1 = sorted(freq_brown , key = freq_brown.__getitem__, reverse = True)
# print(fdist1[0:20])
#
# #prints the most common words with frequency; same result as the previous one with frequency
# print(freq_brown.most_common(20))

# an nltk.ConditionalFreqDist() counts frequencies of pairs.
# When given a list of bigrams, it maps each first word of a bigram
# to a FreqDist over the second words of the bigram.

cfreq_brown_2gram = nltk.ConditionalFreqDist(nltk.bigrams(brown.words()))
# print(cfreq_brown_2gram)

# conditions() in a ConditionalFreqDist are like keys()
# in a dictionary
# print(cfreq_brown_2gram.conditions())

# the cfreq_brown_2gram entry for "my" is a FreqDist.
# print(cfreq_brown_2gram["my"])

# here are the words that can follow after "my".
# We first access the FreqDist associated with "my",
# then the keys in that FreqDist
# print(cfreq_brown_2gram["my"].keys())
#
# # here are the 20 most frequent words to come after "my", with their frequencies
#
# print(cfreq_brown_2gram["my"].most_common(20))


# an nltk.ConditionalProbDist() maps pairs to probabilities.
# One way in which we can do this is by using Maximum Likelihood Estimation (MLE)

cprob_brown_2gram = nltk.ConditionalProbDist(cfreq_brown_2gram, nltk.MLEProbDist)

# This again has conditions() wihch are like dictionary keys
cprob_brown_2gram.conditions()

# Here is what we find for "my": a Maximum Likelihood Estimation-based probability distribution,
# as a MLEProbDist object.

# print(cprob_brown_2gram["my"])

# We can find the words that can come after "my" by using the function samples()

# print(cprob_brown_2gram["my"].samples())


# Here is the probability of a particular pair:

# print(cprob_brown_2gram["my"].prob("own"))


#####

# We can also compute unigram probabilities (probabilities of individual words)

freq_brown_1gram = nltk.FreqDist(brown.words())

len_brown = len(brown.words())


def unigram_prob(word):
    return freq_brown_1gram[ word] / len_brown


#############

# The contents of cprob_brown_2gram, all these probabilities, now form a

# trained bigram language model. The typical use for a language model is

# to ask it for the probabillity of a word sequence

# P(how do you do) = P(how) * P(do|how) * P(you|do) * P(do | you)

prob_sentence = unigram_prob("how") * cprob_brown_2gram["how"].prob("do") * cprob_brown_2gram["do"].prob("you") * cprob_brown_2gram["you"].prob("do")

print(prob_sentence)
# result: 1.5639033871961e-09


