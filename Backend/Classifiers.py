##############################
# Any classifiers should be put here
# TODO
# 1.Naive Bayes
# 2.LSTM
# 3.GRU
# 4.Xgboost
# 5.Logistic regression
# 6.Random forest
# 7.SVC?
# 8.Bagging
# 9.Stacking - cross validation
##############################
import keras.backend as K
import Xgboost as xgboost
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from keras.models import Model
from keras.layers import *


class stacking(object):
    def __init__(self,
                 models,
                 second_model,
                 cv_fold=3,
                 train_size=0.3,
                 mode='Classification',
                 use_proba=False,
                 **kwargs):

        """
        Implement a baseline meta classifier for stacking.

        :param models: list, tuple. List of first level models.
        :param second_model: Second level model.
        :param cv_fold: int. The number of folds used in cross validation training.
        :param train_size: float. The ratio of the training set held out to train first level models
        :param mode: str. Either Classification or Regression
        :param use_proba: boolean. Ignored by mode=Regression. Specify if during training,
                            probabilities will be used as second level meta feature
        :param kwargs:
        """

        self.models = models
        self.n_models = len(models)
        self.sec_model = second_model
        self.use_proba = use_proba
        self.mode = mode

        super().__init__(n_splits=cv_fold, test_size=test_size)

    def fit(self, X, y):
        try:
            X = X.values
            y = y.values
        except:
            None

        X_1, X_2, y_1, y_2 = train_test_split(X, y, train_size=0.5)

        if self.use_proba is not True or mode != 'Classification':
            second_model_features = np.zeros((y_2.shape[0], self.n_models))

            for i, model in enumerate(self.models):
                model.fit(X_1, y_1)
                second_model_features[:, i] = model.predict(X_2)

            self.sec_model.fit(second_model_features, y_2)
        else:
            self.n_classes_ = np.unique(y)

            second_model_features = np.zeros((y.shape[0], self.n_models * self.n_classes))

            for i, model in enumerate(self.models):
                model.fit(X_1, y_1)
                second_model_features[:, i * self.n_classes:(i + 1) * self.n_classes] = model.predict_proba(X_2)

            self.sec_model.fit(second_model_features, y_2)

        return self

    def predict(self, X):
        try:
            X = X.values
        except:
            None

        first_level_prediction = np.empty(shape=(len(X), self.n_models))

        for i, model in enumerate(self.models):
            first_level_prediction[:, i] = model.predict(X)

        final_prediction = self.sec_model.predict(first_level_prediction)

        return final_prediction

    def predict_proba(self, X):
        assert self.mode == 'Classification', 'Regression mode has no predict proba function'

        try:
            X = X.values
        except:
            None

        first_level_prediction = np.empty(shape=(len(X), self.n_models))

        for i, model in enumerate(self.models):
            first_level_prediction[:, i] = model.predict(X)

        final_prediction = self.sec_model.predict_proba(first_level_prediction)

        return final_prediction

class AttentionBlock(Layer):
    def __init__(self, hidden_dim, **kwargs):
        """
        Attention operation for temporal data.
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.

        :param hidden_dim: The dimension of the context vector used to extract relevance word vector.
        :param kwargs:
        """
        self.supports_masking = True
        self.hidden_dim = hidden_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.attention_weight = self.add_weight(shape=(input_shape[2], self.hidden_dim),
                                                name='attention_weight',
                                                initializer='normal',
                                                trainable=True)
        self.attention_bias = self.add_weight(shape=(self.hidden_dim, ),
                                              name='attention_bias',
                                              initializer='zeros',
                                              trainable=True)
        self.latent_vector = self.add_weight(shape=(self.hidden_dim, ),
                                             name='latent_vector',
                                             initializer='normal',
                                             trainable=True)
        # self.latent_copy = K.tile(self.latent_vector, [input_shape[1], 1])

        self.trainable_weights = [self.attention_weight,
                                  self.attention_bias,
                                  self.latent_vector]
        super().build(input_shape)  # Be sure to call this somewhere!

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # To use this, the previous layer has to have a return_sequence option
        u = K.tanh(K.dot(x, self.attention_weight)+self.attention_bias)
        dot_product = K.exp(K.sum(u*self.latent_vector, axis=2))
        alpha = dot_product/K.expand_dims(K.sum(dot_product, axis=1), axis=1)

        if mask is not None:
            mask = K.expand_dims(K.cast(mask, 'float32'), axis=2)
            u *= mask

        weights = K.sum(x*K.expand_dims(alpha, axis=2), axis=1)

        return weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

def BidirectionalRNNAttention(input_length,
                              output_dim,
                              n_words,
                              embedding_dimension,
                              embedding_weights=None,
                              mask=True,
                              RNN_type='LSTM'):
    """
    Implement bidirectional RNN model with attention

    :param input_length: length of each sentence
    :param output_dim: output dimension
    :param n_words: number of words in the vocabulary
    :param embedding_dimension: dimension of the word vectors
    :param embedding_weights: pre-trained word embedding matrix
    :param mask: if mask is set to True, zero will be used as a non-word index
    :param RNN_type: the type ofRNN structure to use, either LSTM or GRU

    :return: model instance
    """

    word_input = Input(shape=(input_length, ), dtype='int32')
    if embedding_weights is not None:
        embedding = Embedding(input_dim=n_words+1,
                              output_dim=embedding_dimension,
                              weights=[embedding_weights],
                              mask_zero=mask)(word_input)
    else:
        embedding = Embedding(input_dim=n_words+1,
                              output_dim=embedding_dimension,
                              mask_zero=mask)(word_input)


    if RNN_type == 'LSTM':
        X = Bidirectional(LSTM(units=64,
                               return_sequences=True))(embedding)
    else:
        X = Bidirectional(GRU(units=64,
                              return_sequences=True))(embedding)

    X = AttentionBlock(64)(X)
    X = Dropout(0.6)(X)
    X = Dense(output_dim, activation='softmax')(X)

    model = Model(inputs=word_input, outputs=X)
    return model

def svc(*args, **kwargs):
    return SVC(*args, **kwargs)

def randomforest(*args, **kwargs):
    return RandomForestClassifier(*args, **kwargs)

def logisticregression(*args, **kwargs):
    return LogisticRegression(*args, **kwargs)

def naivebayes(*args, **kwargs):
    return MultinomialNB(*args, **kwargs)

def bagging(*args, **kwargs):
    return BaggingClassifier(*args, **kwargs)

