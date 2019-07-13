# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple word2vec from scratch with Python
#   2018-FEB
#
# ------------------------------------------------------------------------------+

# --- IMPORT DEPENDENCIES ------------------------------------------------------+

import numpy as np
import re
from collections import defaultdict
from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer

dh_obj = DataHandler()
tk_obj = Tokenizer()

# --- CONSTANTS ----------------------------------------------------------------+
import pandas as pd


class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))  # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # context matrix

        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                # self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))

            print('EPOCH:', i, 'LOSS:', self.loss)
        pass


settings = {}
settings['n'] = 5  # dimension of word embeddings
settings['window_size'] = 2  # context window +/- center word
settings['min_count'] = 0  # minimum word count
settings['epochs'] = 5000  # number of training epochs
settings['neg_samp'] = 10  # number of negative words to use during training
settings['learning_rate'] = 0.01  # learning rate
np.random.seed(0)  # set the seed for reproducibility

# corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

test_corpora_address = "datasets\\input_text_for_test.txt"

test_corpora = pd.read_csv(test_corpora_address)
test_corpora = dh_obj.load_txt_data_as_list(test_corpora_address)
test_corpora = tk_obj.word_tokenizer(test_corpora)
# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, test_corpora)

# train word2vec model
w2v.train(training_data)
