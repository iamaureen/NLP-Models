#https://stackoverflow.com/questions/44088038/create-sentence-row-to-pos-tags-counts-column-matrix-from-a-dataframe

import sys
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

# text = word_tokenize("Maybe if the number is small enough, then it can be a negative number.")
# tag_list = nltk.pos_tag(text)
# print(tag_list)
# tag_fd = nltk.FreqDist(tag for (word, tag) in tag_list)
# print(tag_fd.most_common())
# bigrams = list(nltk.bigrams(text))
# print(bigrams)

import pandas as pd
import os

#read the file
df = pd.read_csv(os.path.join(sys.path[0])+'/pos-tagger-data-example.csv')
df.columns = ['sent', 'tag']
#print(df['sent'])

#create a function tok_and_tag that does word_tokenize and pos_tag in a chained manner:
tok_and_tag = lambda x: nltk.pos_tag(word_tokenize(x))
#get the first sentence
# print(df['sent'][0])
# print(tok_and_tag(df['sent'][0]))

#Then, we can use df.apply to tokenize and tag the sentence column of the dataframe:
#df['sent'].apply(tok_and_tag)

#lowercase of the sentence
df['lower_sent'] = df['sent'].apply(str.lower)
#apply the function to get the tag
df['tagged_sent'] = df['lower_sent'].apply(tok_and_tag)


tokens, tags = zip(*nltk.chain(*df['tagged_sent'].tolist()))

print(tags)

possible_tags = sorted(set(tags))

#initialize possible_tags_counter with 0
possible_tags_counter = Counter({p:0 for p in possible_tags})
print(possible_tags_counter)

#iterate through each tagged sentence and get the counts of POS:
df['pos_counts'] = df['tagged_sent'].apply(lambda x: Counter(list(zip(*x))[1]))
print(df['pos_counts'])

#add in the POS that don't appears in the sentence with 0 counts:
def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

df['pos_counts_with_zero'] = df['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))

df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])
print(df['pos_counts_with_zero'])

#flatten the values into the list:
df['sent_vector'] = df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])
print(df['sent_vector'])

#create a dataframe to store the matrix
df2 = pd.DataFrame(df['sent_vector'].tolist())
df2.columns = sorted(possible_tags)
df2.to_csv(os.path.join(sys.path[0])+'/POS-tagger-output.csv', index = False, sep=',', encoding='utf-8')
print(df2)