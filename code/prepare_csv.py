#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:12:10 2020

@author: muazzam
"""

import os
from os import walk
import pandas as pd

# read the corpus
# each review is stored as a separate text file in its respective folder
# folders name can serve as labels, but we already have labels stored in separate file

data = {}
path = '/Research/Sentiment Analysis/Sentiment analysis and subjectivity detection in Arabic documents/Corpus/ABSA7/';
for (dirpath, dirnames, filenames) in walk(path):
    for f in filenames:
        if f.endswith('txt'):
            with open(os.path.join(dirpath, f)) as file:
                text = ''
                for line in file:
                    text += line.strip()
                data[f] = text
print(len(data))

# prepare text samples and their labels
path = '/Research/Sentiment Analysis/Sentiment analysis and subjectivity detection in Arabic documents/Data/'
labels = pd.read_csv(os.path.join(path, 'absa7_labels.txt'))
labels['review'] = 'review'

# add the reviews to the labels dataframe
# instead of adding 7 label columns to the data dataframe
# it is easier to get the review from the data dataframe and add it to labels
for i in labels.index:
    labels.loc[i, 'review'] = data[labels.loc[i, 'filename']]

# write the data as a csv file
labels.to_csv(path+'absa7_data.csv')