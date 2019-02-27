# http://www.katrinerk.com/courses/python-worksheets/language-models-in-python
# Katrin Erk Oct 07
# Updated Feb 11
#
# Word bigrams are just pairs of words.
# In the sentence "I went to the beach"
# the bigrams are:
#    I went
#    went to
#    to the
#    the beach
#
# Having counts of English bigrams from a very large text corpus
# can be useful for a number of purposes.
#
# for example for spelling correction:
# If I had mistyped the sentence as "I went to beach"
# then I might be able to find the error by seeing that
# the bigram "to beach" has a very low count, and
# "to the", "to a", and "the beach" have much larger counts.
#
# This program counts all word bigrams in a given text file
#
# usage:
# python3 count_bigrams.py <filename>
#
# <filename> is a text file.

import string
import sys

# complain if we didn't get a filename
# as a command line argument
if len(sys.argv) < 2:
    print("Please enter the name of a corpus file as a command line argument.")
    sys.exit()

# try opening file
# If the file doesn't exist, catch the error
try:
    f = open(sys.argv[1])
except IOError:
    print("Sorry, I could not find the file", sys.argv[1])
    print("Please try again.")
    sys.exit()

# read the contents of the whole file into ''filecontents''
filecontents = f.read()

# count bigrams
bigrams = {}
words_punct = filecontents.split()
# strip all punctuation at the beginning and end of words, and
# convert all words to lowercase.
# The following is a Python list comprehension. It is a command that transforms a list,
# here words_punct, into another list.
words = [w.strip(string.punctuation).lower() for w in words_punct]

# add special START, END tokens
words = ["START"] + words + ["END"]

for index, word in enumerate(words):
    if index < len(words) - 1:
        # we only look at indices up to the
        # next-to-last word, as this is
        # the last one at which a bigram starts
        w1 = words[index]
        w2 = words[index + 1]
        # bigram is a tuple,
        # like a list, but fixed.
        # Tuples can be keys in a dictionary
        bigram = (w1, w2)

        if bigram in bigrams:
            bigrams[bigram] = bigrams[bigram] + 1
        else:
            bigrams[bigram] = 1
        # or, more simply, like this:
        # bigrams[bigram] = bigrams.get(bigram, 0) + 1

# sort bigrams by their counts
sorted_bigrams = sorted(bigrams.items(), key=lambda pair: pair[1], reverse=True)

for bigram, count in sorted_bigrams:
    print(bigram, ":", count)