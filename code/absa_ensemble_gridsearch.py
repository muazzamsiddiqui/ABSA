#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:10:20 2020

@author: muazzam
"""
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import LabelSpacePartitioningClassifier

from sklearn.svm import SVC

# this is is not fully annotated ABSA data. it comes with aspect categorization only
# sentiment annotation will be added to the aspects in the future
# this is a subset of the corpus prepared for the "Sentiment Analysis and Subjectivity Detection in Arabic Documents" project
# if you use this data in your research please cite the following paper
# Muazzam Ahmed Siddiqui, Mohamed Yehia Dahab, and Omar Abdullah Batarfi. 2015. Building A Sentiment Analysis Corpus With Multifaceted Hierarchical Annotation. International Journal of Computational Linguistics (IJCL) (2015).

rawdata = pd.read_csv('../data/absa7_data.csv')

reviews = rawdata['review'].values
possible_labels = ["HOTEL#CLEANLINESS", "HOTEL#COMFORT", "HOTEL#GENERAL", "HOTEL#PRICE", "LOCATION#GENERAL", "SERVICE#GENERAL", "STAFF#GENERAL"]
labels = rawdata[possible_labels].values

df_train, df_test, Ytrain, Ytest = train_test_split(reviews, labels, test_size=0.2)

tfidf = TfidfVectorizer()
Xtrain = tfidf.fit_transform(df_train)
Xtest = tfidf.transform(df_test)

parameters = {
    'classifier': [BinaryRelevance(), LabelPowerset(), ClassifierChain()],
    #'classifier__classifier': [RandomForestClassifier()],
    'classifier__classifier': [SVC(kernel='linear', gamma='auto')],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}

clf = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring = 'f1_micro')
clf.fit(Xtrain, Ytrain)

print (clf.best_params_, clf.best_score_)

prediction = clf.predict(Xtest)

print ("Hamming Loss = " + str(metrics.hamming_loss(Ytest, prediction)))

print("Accuracy = " + str(metrics.accuracy_score(Ytest, prediction)))

print("Micro Precision = " + str(metrics.precision_score(Ytest, prediction, average='micro')))

print("Micro Recall = " + str(metrics.recall_score(Ytest, prediction, average='micro')))

print("Micro F1 = " + str(metrics.f1_score(Ytest, prediction, average='micro')))

print("Macro Precision = " + str(metrics.precision_score(Ytest, prediction, average='macro')))

print("Macro Recall = " + str(metrics.recall_score(Ytest, prediction, average='macro')))

print("Macro F1 = " + str(metrics.f1_score(Ytest, prediction, average='macro')))
