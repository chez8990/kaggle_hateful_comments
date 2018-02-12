import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import *
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import *
from sklearn.multiclass import OneVsRestClassifier

# define meta features
def count_punctuations()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ([SnowballStemmer('english').stem(w) for w in analyzer(doc)])


train = pd.read_csv('train_cleaned.csv', header=0, encoding='iso-8859-1')
test = pd.read_csv('test_cleaned.csv', header=0, encoding='iso-8859-1')

scv = StemmedCountVectorizer(stop_words='english')
tfidf = TfidfTransformer()
X = scv.fit_transform(train['w/o_punctuation'])
X_tfidf = tfidf.fit_transform(X)

y = train[['toxic', 'severe_toxic', 'obscene', 'threat',
           'insult', 'identity_hate']].values

X_train, X_test, \
X_ttrain, X_ttest, \
X_ltrain, X_ltest, \
y_train, y_test = train_test_split(X, X_tfidf, X_lsa, y)


# fitting a benchmark classifier
nb = OneVsRestClassifier(MultinomialNB())
nb.fit(X_train, y_train)

lr = OneVsRestClassifier(LogisticRegression())
lr.fit(X_ltrain, y_train)

nb_tfidf = OneVsRestClassifier(MultinomialNB())
nb.fit(X_ttrain, y_train)

# cross validation
kf = KFold(n_splits=10, shuffle=True)
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb.fit(X_train, y_train)

    scores.append(nb.predict_proba(X))


# prediction for submission
test['w/o_punctuation'] = test['w/o_punctuation'].fillna(' ')

X_predict = scv.transform(test['w/o_punctuation'])
prediction = pd.DataFrame(nb.predict_proba(X_predict),
                          columns=['toxic', 'severe_toxic', 'obscene', 'threat',
                                   'insult', 'identity_hate'])

prediction = pd.concat([test, prediction], axis=1)
prediction.drop(['comment_text', 'w/o_punctuation'], axis=1, inplace=True)
prediction.to_csv('submission_benchmark.csv', index=None)