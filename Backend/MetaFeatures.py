import re
import string
from nltk import pos_tag, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
with open('..\data\swear_words.txt', 'r') as f:
    swear_words = [word.strip() for word in f.readlines()]

class SentenceStructure(str, SentimentIntensityAnalyzer):
    def __init__(self, sentence):
        self.sentence = sentence
        self.lower_sentence = sentence.lower()
        self.Tokenized()
        self.SentenceLength()
        self.POS()
        self.sentiment()
        self._run_letters()
        self._run_words()

        super(SentenceStructure, self).__init__()

    def Tokenized(self):
        self.tokenized = word_tokenize(self.sentence)

    def SentenceLength(self):
        self.length = len(self.tokenized)

    def POS(self):
        self.pos = pos_tag(self.tokenized)

    def sentiment(self):
        self.sent_dict = self.polarity_scores(self.sentence)

    def _run_letters(self):
        ###########
        # single loop over all the letters in the sentence
        ###########

        self.punc_count = 0
        self.n_capital = 0

        for letter in self.sentence:
            if letter in punc:
                self.punc_count += 1
            if letter.isupper():
                self.n_capital += 1

    def _run_words(self):
        ###########
        # single loop over all the words in tokenized
        ###########

        self.n_names = 0
        self.n_swear = 0

        for i, word in enumerate(self.tokenized):
            if word.lower() in swear_words:
                self.n_swear += 1
            if self.pos[i][1] == 'NNP':
                self.n_names += 1

def SentenceAnalysis(sentence):
    structure = SentenceStructure(sentence)
    numerical_attributes = {'n_names':s.n_names,
                            'n_swear':s.n_swear,
                            'n_capital':s.n_capital,
                            'punc_count':s.punc_count,
                            **s.sent_dict
                            }

    return numerical_attributes