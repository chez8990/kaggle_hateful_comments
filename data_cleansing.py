import pandas as pd
import numpy as np
import nltk
import string
import re
from string import digits
from nltk.corpus import words
from nltk import word_tokenize
# import snowballstemmer
from collections import Counter



train = pd.read_csv('train.csv', header=0)
test = pd.read_csv('test.csv', header=0)
train['comment_text'] = train['comment_text'].apply(lambda x: x.strip())
test['comment_text'] = test['comment_text'].apply(lambda x: str(x).strip())



# Exclusion List
excl_df = pd.read_csv("exclusion_list.csv")
exclusion_list = excl_df['Header'].values

# Text cleaning

# remove useless punctuations
remain = r'\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\@|\[|\||\]|\^|\_|\`|\{|\||\}|\~|'
remain2 = r'\!|\?|“|”|\n'
                           
# cleaning in general
def cleaning(text):
    text = str(text)
    text = text.lower()
    
    # TODO: determine potential dirty word
    
    #remove_quotations = text.strip('"').strip()
    remove_quotations = str.replace(text, r'"', '')
    remove_punctuations = re.sub(remain, '', remove_quotations)
    remove_punctuations = re.sub(remain2, ' ', remove_punctuations)
    
    # remove the multiple space
    remove_mult_space = re.sub(' +', ' ', remove_punctuations)
    
    # Remove IP Address
    remove_digits = str.maketrans('', '', digits)
    remove_number = remove_mult_space.translate(remove_digits)
    remove_long_word = re.sub(r'\b\w{15,}\b', '', remove_number)
    
    # Remove the unknown word
    #word_list = set(words.words())
    #" ".join(w for w in nltk.wordpunct_tokenize(remove_long_word) if w.lower() in word_list or not w.isalpha())
    
    # Remove exclusion words
    remove_excl = remove_long_word
    #for word in list(exclusion_list):
    #    remove_excl = str.replace(remove_excl, word, '')
    
    # remove the multiple space
    remove_mult_space = re.sub(' +', ' ', remove_excl)
    
    return remove_mult_space

def preprocessing(data):
    # stemmer = snowballstemmer.EnglishStemmer()
    #stopword = set(stopwords.words('english'))
    print("cleansing...")
    data['w/o_punctuation'] = data['comment_text'].apply(cleaning)
    print("excluding...")
    i = 1
    for excl_word in exclusion_list:
        data['w/o_punctuation'] = data['w/o_punctuation'].str.replace(excl_word, '')
        print(str(i) + " out of " + str(len(exclusion_list)))
        i = i + 1

    return data


train = preprocessing(train)
test = preprocessing(test)

train.to_csv('train_cleaned.csv', index=None)
test.to_csv('test_cleaned.csv', index=None)

arr = np.append(train['w/o_punctuation'].values, test['w/o_punctuation'].values)
dict_df = pd.DataFrame.from_dict(Counter(" ".join(arr).split()), orient="index").sort_values(by=[0])
dict_df.to_csv("word_count.csv", index=True, encoding="utf-8")
