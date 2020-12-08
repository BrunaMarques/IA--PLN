import json
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from data import extract_data


ngrama = 2
k = 100

X_sentences, Y_sentences = extract_data()
print("tfidf")
vectorizer = TfidfVectorizer(ngram_range=(1, ngrama), stop_words='english')
X_sentences = vectorizer.fit_transform(X_sentences)


print("\nClassificador SVM: ")
param_grid = [{'C': [1, 10, 100, 1000], 'multi_class':['ovr', 'crammer_singer'], 'penalty':['l1', 'l2'], 'fit_intercept':['True', 'False'], 'max_iter':[1000, 5000]}]
clf = GridSearchCV(LinearSVC(), param_grid, cv = 10, n_jobs=-1) 
clf = LinearSVC(random_state=0, tol=1e-5)
clf = clf.fit(X_sentences, Y_sentences)
print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))

#################### Random Forest ##########################
print("\nClassificador Random Forest:...")
param_grid =  [{'n_estimators' :[100, 300, 500],'max_depth' :[5, 8, 15],'min_samples_split' :[2, 5, 10],'min_samples_leaf':[1, 2, 5]}]
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv = 10, n_jobs=-1) 
clf = clf.fit(X_sentences, Y_sentences)

print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))

################### Naive #############################
print("\nClassificador Naive Bayes:")
clf = MultinomialNB()

clf = clf.fit(X_sentences, Y_sentences)

print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))




