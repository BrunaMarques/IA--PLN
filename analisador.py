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
from data2 import extract_data


ngrama = 1
k = 100

X_sentences, X_prev, X_next, X_pos, Y_sentences = extract_data()
print("tfidf")
vectorizer = TfidfVectorizer(ngram_range=(1, ngrama))
X_sentences = vectorizer.fit_transform(X_sentences)
X_prev = vectorizer.transform(X_prev)
X_next = vectorizer.transform(X_next)

print(len(vectorizer.get_feature_names()))

print("chi-quadrado")
selector = SelectKBest(chi2, k=k)
X_sentences = selector.fit_transform(X_sentences, Y_sentences)
X_prev = selector.transform(X_prev)
X_next = selector.transform(X_next)

print("adicionando anterior e posterior")
X_sentences = hstack([X_sentences, X_prev, X_next, np.expand_dims(np.array(X_pos), -1)])


print("\nClassificador SVM: ")
# undersample = RandomUnderSampler(sampling_strategy='all')
# #undersample = RandomUnderSampler(sampling_strategy='majority')
# param_grid = [{'class_weight': ['balanced'],'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
# clf = GridSearchCV(SVC(), param_grid, cv = 10) 
# clf = SVC(random_state=0, tol=1e-5)
clf = NuSVC()
# clf = LinearSVC(random_state=0, tol=1e-5)
print("AAA")
# X_sentences, Y_sentences = undersample.fit_resample(X_sentences, Y_sentences)
clf = clf.fit(X_sentences, Y_sentences)
print("Predicão...")
pred = cross_val_predict(clf, X_sentences, Y_sentences)

print("Classification_report:")
print(classification_report(Y_sentences, pred,  zero_division = 0))
print("")
print(confusion_matrix(Y_sentences, pred))

# print('melhores parametros: ', clf.best_params_)
#################### Random Forest ##########################
# print("\nClassificador Random Forest:...")
# param_grid =  [{'class_weight': ['balanced_subsample'], 'max_depth': [20], 'max_features': ['log2'], 'n_estimators': [100]}]
# #undersample = RandomUnderSampler(sampling_strategy='all')
# undersample = RandomUnderSampler(sampling_strategy='majority')
# X_sentences, Y_sentences = undersample.fit_resample(X_sentences, Y_sentences)
# clf = GridSearchCV(RandomForestClassifier(), param_grid, cv = 10) 
# clf = clf.fit(X_sentences, Y_sentences)

# print("Predicão...")
# pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)

# print("Classification_report:")
# print(classification_report(Y_sentences, pred,  zero_division = 0))
# print("")
# print(confusion_matrix(Y_sentences, pred))

##################### Naive #############################
# print("\nClassificador Naive Bayes:")
# param_grid =  [{'float': [1.0,2.0, 0.5]}]
# #undersample = RandomUnderSampler(sampling_strategy='all')
# #undersample = RandomUnderSampler(sampling_strategy='majority')
# #X_sentences, Y_sentences = undersample.fit_resample(X_sentences, Y_sentences)
# # clf = GridSearchCV(MultinomialNB(), param_grid, cv = 10) 
# clf = ComplementNB()

# clf = clf.fit(X_sentences.toarray(), Y_sentences)

# print("Predicão...")
# pred = cross_val_predict(clf, X_sentences.toarray(), Y_sentences, cv=10)

# print("Classification_report:")
# print(classification_report(Y_sentences, pred,  zero_division = 0))
# print("")
# print(confusion_matrix(Y_sentences, pred))


# #################### MLP ###############################
# print("\nClassificador MLP:")
# param_grid = [{'learning_rate' :['constant', 'invscaling', 'adaptive'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [0.0001, 0.05]}]
# undersample = RandomUnderSampler(sampling_strategy='all')
# #undersample = RandomUnderSampler(sampling_strategy='majority')
# X_sentences, Y_sentences = undersample.fit_resample(X_sentences, Y_sentences)
# clf = GridSearchCV(MLPClassifier(), param_grid, cv = 10)
# clf = clf.fit(X_sentences, Y_sentences)

# print("Predicão...")
# pred = cross_val_predict(clf, X_sentences, Y_sentences, cv=10)

# print("Classification_report:")
# print(classification_report(Y_sentences, pred,  zero_division = 0))
# print("")
# print(confusion_matrix(Y_sentences, pred))
# # print(corpus)



