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
    sentences = []
    labels = []
    data = pd.read_csv('IMDBDataset.csv', usecols=['review', 'sentiment'])
    print(len(data))
    for i in range(len(data)):    
        sentences.append(cleanhtml(data.values[i][0]))
        labels.append(data.values[i][1])
    return sentences, labels

extract_data()
