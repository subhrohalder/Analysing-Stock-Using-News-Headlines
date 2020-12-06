#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:49:26 2020

@author: subhrohalder
"""

import pandas as pd

#importng the dat dataset
news_df = pd.read_csv('Data.csv', encoding= 'unicode_escape')

#checking if the datset is imbalanced or?
news_df['Label'].value_counts()

#train test split
train_set = news_df[news_df['Date']<'20150101']
test_set= news_df[news_df['Date']>'20141231']

#removing the punctuation from training data set
train_set_clean = train_set.iloc[:,2:27]
train_set_clean.replace("[^a-zA-z]"," ",regex=True,inplace=True)

#adding all the columns
for i in train_set_clean.columns.values:
   train_set_clean[i] = train_set_clean[i].str.lower()
   
headlines = []

for i in range (0,len(train_set_clean.index)):
    headlines.append(' '.join(str(i) for i in train_set_clean.iloc[i,0:25]))
    
headlines[0]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

#vectorization
count_vectorizer = CountVectorizer(ngram_range=(3,3))
traindataset = count_vectorizer.fit_transform(headlines)

#fitting
rf_clf = RandomForestClassifier(n_estimators=200,criterion='entropy')
rf_clf.fit(traindataset,train_set['Label'])

#testing
test_set_clean = test_set.iloc[:,2:27]
test_set_clean.replace("[^a-zA-z]"," ",regex=True,inplace=True)


for i in test_set_clean.columns.values:
   test_set_clean[i] = test_set_clean[i].str.lower()
   
test_headlines = []

for i in range (0,len(test_set_clean.index)):
    test_headlines.append(' '.join(str(i) for i in test_set_clean.iloc[i,0:25]))
    
testdataset = count_vectorizer.transform(test_headlines)
predictions = rf_clf.predict(testdataset)

predictions[0]


from sklearn.metrics import classification_report
print(classification_report(test_set['Label'], predictions))