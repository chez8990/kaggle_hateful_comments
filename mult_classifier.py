import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import *
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import LatentDirichletAllocation
# define meta features
#def count_punctuations()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ([SnowballStemmer('english').stem(w) for w in analyzer(doc)])

    

train = pd.read_csv('train_cleaned.csv', header=0, encoding='iso-8859-1')
test = pd.read_csv('test_cleaned.csv', header=0, encoding='iso-8859-1')

test['w/o_punctuation'] = test['w/o_punctuation'].fillna(' ')



scv = StemmedCountVectorizer(stop_words='english')
tfidf = TfidfTransformer()

X_all = scv.fit_transform(np.append(train['w/o_punctuation'].values, test['w/o_punctuation'].values))
X = scv.transform(train['w/o_punctuation'])
X = tfidf.fit_transform(X)


#lda = LatentDirichletAllocation(n_components=20)
#X = lda.fit_transform(X)


y = train[['toxic', 'severe_toxic', 'obscene', 'threat',
           'insult', 'identity_hate']].values

X_train, X_test, \
y_train, y_test = train_test_split(X, y)

X_predict = scv.transform(test['w/o_punctuation'])
X_predict = tfidf.transform(X_predict)
#X_predict = lda.transform(X_predict)

print("fitting..")
# fitting a benchmark classifier
#clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, max_depth=3)
clf = LogisticRegression(class_weight='balanced')
#clf = GradientBoostingClassifier(max_depth=3)
clf.fit(X, train['toxic'].values)
y_pred = clf.predict(X)
print(confusion_matrix(train['toxic'].values, y_pred))
print(clf.score(X, train['toxic'].values))

clf2 = LogisticRegression(class_weight='balanced')
# clf2 = GradientBoostingClassifier(max_depth=3)
clf2.fit(X, train['severe_toxic'].values)
y_pred2 = clf2.predict(X)
print(confusion_matrix(train['severe_toxic'].values, y_pred2))
print(clf2.score(X, train['severe_toxic'].values))

clf3 = LogisticRegression(class_weight='balanced')
# clf3 = GradientBoostingClassifier(max_depth=3)
clf3.fit(X, train['obscene'].values)
y_pred3 = clf3.predict(X)
print(confusion_matrix(train['obscene'].values, y_pred3))
print(clf3.score(X, train['obscene'].values))

clf4 = LogisticRegression(class_weight='balanced')
# clf4 = GradientBoostingClassifier(max_depth=3)
clf4.fit(X, train['threat'].values)
y_pred4 = clf4.predict(X)
print(confusion_matrix(train['threat'].values, y_pred4))
print(clf4.score(X, train['threat'].values))

clf5 = LogisticRegression(class_weight='balanced')
# clf5 = GradientBoostingClassifier(max_depth=3)
clf5.fit(X, train['insult'].values)
y_pred5 = clf5.predict(X)
print(confusion_matrix(train['insult'].values, y_pred5))
print(clf5.score(X, train['insult'].values))

clf6 = LogisticRegression(class_weight='balanced')
# clf6 = GradientBoostingClassifier(max_depth=3)
clf6.fit(X, train['identity_hate'].values)
y_pred6 = clf6.predict(X)
print(confusion_matrix(train['identity_hate'].values, y_pred6))
print(clf6.score(X, train['identity_hate'].values))




# prediction for submission


print("Predicting..")


y_test_pred1 = clf.predict_proba(X_predict)
y_test_pred2 = clf2.predict_proba(X_predict)
y_test_pred3 = clf3.predict_proba(X_predict)
y_test_pred4 = clf4.predict_proba(X_predict)
y_test_pred5 = clf5.predict_proba(X_predict)
y_test_pred6 = clf6.predict_proba(X_predict)


prediction = pd.DataFrame({'toxic': y_test_pred1[:,1], 'severe_toxic': y_test_pred2[:,1], 'obscene': y_test_pred3[:,1], 
                         'threat': y_test_pred4[:,1], 'insult': y_test_pred5[:,1], 'identity_hate': y_test_pred6[:,1]})

prediction = pd.concat([test, prediction], axis=1)
prediction.drop(['comment_text', 'w/o_punctuation'], axis=1, inplace=True)
print("Output..")
prediction.to_csv('submission.csv', index=None)
