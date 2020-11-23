import csv
import numpy as np
import pandas as pd
import emojis
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re

def cleanhtml(raw_html):   
    cleanr = re.compile('<.*?>')   
    cleantext = re.sub(cleanr, '', raw_html)  
    return cleantext

def extract_data():
    aux = []
    count = 0
    sentences = []
    labels = []
    sentences_pos = []
    sentences_previous = []
    sentences_next = []
    data = pd.read_csv('IMDBDataset.csv', usecols=['review', 'sentiment'])
    print(len(data))
    for i in range(len(data)): 
        sentences.append(data.values[i][0])
        labels.append(data.values[i][1])

    for i in range(len(sentences)):
        sentences_pos.append(i)
        if i - 1 >= 0:
            sentences_previous.append(sentences[i - 1])
        else:
            sentences_previous.append("")

        if i + 1 < len(sentences):
            sentences_next.append(sentences[i + 1])
        else:
            sentences_next.append("")
    p, n= 0, 0
    for i in labels:
        if i == 'positive':
            p+=1
        else:
            n+= 1
    print(p,n)
    print(len(sentences))
    return sentences, sentences_previous, sentences_next, sentences_pos, labels

extract_data()
