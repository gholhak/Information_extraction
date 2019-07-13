import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame
from data_handler.data_utils import DataHandler
from concept_extraction.features_extract import CoOccurrence
from concept_extraction.FarsiTokenizer import Tokenizer
import matplotlib.pyplot as plt

config = {}
config['w_size'] = 2
co_occ_obj = CoOccurrence(config)
tk_obj = Tokenizer()
dh_obj = DataHandler()


class SimpleNeuralNetwork:
    def __init__(self):

        self.learning_rate = 0.00001
        self.window_size = 2
        self.epoch = 500

        """
        networks' configuration
        """
        self.setting = {'learning_rate': self.learning_rate,
                        'window_size': self.window_size, 'epoch': self.epoch}
        '''
        weights initialization
        '''

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def generate_training_data(self, co_mat, co_occ_dictionary, corpus, encoding=1):
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                # w_target = self.word2onehot(sentence[i])
                w_target = (co_occ_obj.get_vector(sentence[i]))

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and sent_len - 1 >= j >= 0:
                        w_context.append(co_occ_obj.get_vector(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

        # training_data = []
        # jj = 0
        #
        # if encoding == 3:
        #     co_mat = co_occ_obj.binerize_co_occurrence(co_mat)
        # elif encoding == 2:
        #     pass
        #
        # for item in co_occ_dictionary:
        #     temp_data = []
        #     oneHotdata = []
        #     for key in item:
        #         one_hot_w_target = self.word2onehot(key)
        #         center_word = DataFrame.as_matrix(co_mat[jj].loc[key, :])
        #         center_word[np.isnan(center_word)] = 0
        #         center_word = list(center_word)
        #         center_context = []
        #         for sub_item in item[key]:
        #             temp = DataFrame.as_matrix(co_mat[jj].loc[sub_item, :])
        #             temp[np.isnan(temp)] = 0
        #             temp = list(temp)
        #             one_hot_context = self.word2onehot(sub_item)
        #             center_context.append(temp)
        #         temp_data.append([center_word, center_context])
        #         oneHotdata.append([one_hot_w_target, center_context])
        #     jj = jj + 1
        # if encoding == 1:
        #     training_data.append(oneHotdata)
        # else:
        #     training_data.append(temp_data)
        # return training_data

    '''
    sigmoid activation function
    '''

    @staticmethod
    def activation_function(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def word_count_per_document(corpus):

        uni_final = []
        for sent in corpus:
            uni = []
            for term in sent:
                if term not in uni:
                    uni.append(term)
            uni_final.append(uni)
        return uni_final

    def feed_forward(self, w1, w2, center_word):
        y1 = np.dot(w1.T, center_word)
        y2 = np.dot(w2.T, y1)
        y_pred = self.activation_function(y2)
        return y_pred, y1, y2

    def back_propagation(self, error, output_vector1, hidden_to_output_weights,
                         input_to_hidden_weights, center_word):

        """calculates the derivatives of error with respect to weights"""
        der_w2 = np.outer(output_vector1, error)
        der_w1 = np.outer(center_word, np.dot(hidden_to_output_weights, error.T))

        '''gradient descent'''
        self.input_to_hidden_weights = input_to_hidden_weights - (self.setting['learning_rate'] * der_w1)
        self.hidden_to_output_weights = hidden_to_output_weights - (self.setting['learning_rate'] * der_w2)
        pass

    def train_network(self, training_data, complete_context):

        unique_terms = self.word_count_per_document(complete_context)
        dim = np.power(self.setting['window_size'], 2) + 1
        # dim = 2

        np.random.seed(100)
        self.input_to_hidden_weights = np.random.uniform(-0.8, 0.8, (len(unique_terms[0]), dim))
        self.hidden_to_output_weights = np.random.uniform(-0.8, 0.8, (dim, len(unique_terms[0])))
        epch_holder = []
        loss_holder = []
        for i in range(0, self.setting['epoch']):
            self.loss = 0
            for center_word, center_context in training_data:
                y_pred, output_vector1, output_vector2 = self.feed_forward(self.input_to_hidden_weights,
                                                                           self.hidden_to_output_weights,
                                                                           center_word)

                error = np.sum([np.subtract(y_pred, word) for word in center_context],
                               axis=0)

                self.back_propagation(error, output_vector1, self.hidden_to_output_weights,
                                      self.input_to_hidden_weights, center_word)

                self.loss += -np.sum([output_vector2[word == 1] for word in center_context]) + len(
                    center_context) * np.log(np.sum(np.exp(output_vector2)))
            print('EPOCH:', i, 'LOSS:', self.loss)
            loss_holder.append(self.loss)
            epch_holder.append(i)
            pass
        return epch_holder, loss_holder


def main():
    ann_obj = SimpleNeuralNetwork()

    main_corpora_address = 'datasets\\raw_ner_data.csv'
    test_corpora_address = 'datasets\\input_text_for_test.txt'
    corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]

    main_corpora = pd.read_csv(main_corpora_address)
    test_corpora = dh_obj.load_txt_data_as_list(test_corpora_address)

    # test_corpora = tk_obj.sent_tokenizer(test_corpora)
    test_corpora = tk_obj.word_tokenizer(test_corpora)

    decomposed_context, complete_context = tk_obj.document_extraction(test_corpora, test=True)

    co_mat, co_occ_dictionary, co_occ_dictionary_unique, unique_labels = co_occ_obj.build_co_occurrence_matrix(
        complete_context)
    # vectors = co_occ_obj.build_vector_from_co_mat(co_mat, co_occ_dictionary)

    # co_occ_dictionary_unique testing
    # please specify the encoding strategy, default is 1
    # one_hot encoding = 1
    # co_occurrence matrix default values = 2
    # co_occurrence matrix in binary format = 3
    training_set = ann_obj.generate_training_data(co_mat, co_occ_dictionary_unique, complete_context, encoding=1)
    epoch, loss = ann_obj.train_network(training_set, complete_context)
    plt.plot(epoch, loss)
    plt.show()


if __name__ == '__main__':
    main()
