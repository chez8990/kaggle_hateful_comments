import pandas as pd
import nltk
import string
import re
from nltk import word_tokenize

train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
train['comment_text'] = train['comment_text'].apply(lambda x: x.strip())
test['comment_text'] = test['comment_text'].apply(lambda x: str(x).strip())

# Text cleaning

# remove useless punctuations
remain = r'\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\@|\[|\||\]|\^|\_|\`|\{|\||\}|\~'

# cleaning in general
def cleaning(text):
    text = str(text)
    remove_quotations = text.strip('"').strip()
    remove_new_line = re.sub(re.compile(r'\n'), '', remove_quotations)
    remove_punctuations = re.sub(remain, '', remove_new_line)

    return remove_punctuations

def preprocessing(data):
    stemmer = SnowballStemmer('english')
    stopword = set(stopwords.words('english'))
    data['w/o_punctuation'] = data['comment_text'].apply(cleaning)


preprocessing(train)
preprocessing(test)

train.to_csv('train_cleaned.csv', index=None)
test.to_csv('test_cleaned.csv', index=None)