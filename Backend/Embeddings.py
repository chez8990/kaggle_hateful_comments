##############################
# Any Embedding related code should be put here
# TODO
# 1. CountVectorizer (with different grams, stemming, stopwords)
# 2. TfidfVectorizer (with different grams, stemming, stopwords)
# 3. Word2Vec
# 4. Glove
# 5. FastText
# 6. LSA, LDA on CountVectorizer
# 7. PCA, TSNE on all embeddings except CountVectorizer
##############################
import numpy as np
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem import SnowballStemmer

class StemmedCountVectorizer(CountVectorizer):
    #########################
    # Countvectorizer subclass to include a stemming analyzer
    #########################

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ([SnowballStemmer('english').stem(w) for w in analyzer(doc)])

class MetaPipeline(Pipeline):
    #########################
    # Pipeline that does not require the last object
    # in the steps to be an estimator
    #########################

    def fit(self, X, y=None, **fit_params):
        X_next = X
        for _, step in self.steps:
            try:
                X_next = step.fit_transform(X_next)
            except:
                step.fit(X_next, y)

        return self

def WeightedMeanEmbedding(sequence_matrix,
                          dimensions,
                          embedding_lookup,
                          weights,
                          pad=None,):

    #########################
    # Provides an embedding that has a fixed dimension
    # by means of weighted average
    #########################

    assert len(weights.shape) == 2

    if pad is not None:
        assert type(pad)==int, 'Provide an integer value for padding'
        sequence_matrix = np.pad(sequence_matrix, pad_widht=(0,pad), mode='constant')

    n_samples, max_length = sequence_matrix.shape

    weights = np.repeat(weights[:, :, np.newaxis], axis=2)

    embedding_matrix = np.zeros(shape=(n_samples, max_length, dimensions))

    for index in embedding_lookup.keys(:
        places = np.argwhere(sequence_matrix==index)
        vector = embdding_lookup[index]

        embedding_matrix[places, :] = vector

    return np.mean(embedding_matrix*weights, axis=1)

def pca(*args, **kwargs):
    return PCA(*args, **kwargs)

def svd(*args, **kwargs):
    return TruncatedSVD(*args, **kwargs)

def lda(*args, **kwargs):
    return LatentDirichletAllocation(*args, **kwargs)

def tsne(*args, **kwargs):
    return TSNE(*args, **kwargs)

def w2v(*args, **kwargs):
    return Word2Vec(*args, **kwargs)

def fasttext(*args, **kwargs):
    return FastText(*args, **kwargs)