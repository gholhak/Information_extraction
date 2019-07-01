from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame
from data_handler.data_utils import DataHandler
from concept_extraction.features_extract import CoOccurrence
from concept_extraction.FarsiTokenizer import Tokenizer

config = {}
config['w_size'] = 2
co_occ_obj = CoOccurrence(config)
tk_obj = Tokenizer()
dh_obj = DataHandler()


class SimpleNeuralNetwork:
    def __init__(self):

        self.num_input_nodes = 4
        self.num_hidden_nodes = 4
        self.num_output_nodes = 2
        self.learning_rate = 0.01
        self.window_size = 2
        self.epoch = 10
        """
        networks' configuration
        """
        self.setting = {'num_input_nodes': self.num_input_nodes, 'num_hidden_nodes': self.num_hidden_nodes,
                        'num_output_nodes': self.num_output_nodes, 'learning_rate': self.learning_rate,
                        'window_size': self.window_size, 'epoch': self.epoch}
        '''
        weights initialization
        '''

    @staticmethod
    def generate_training_data(co_mat, dict_mat):

        training_data = []
        jj = 0
        for item in dict_mat:
            temp_data = []
            for key in item:
                center_word = DataFrame.as_matrix(co_mat[jj].loc[key, :])
                center_word[np.isnan(center_word)] = 0
                center_word = list(center_word)
                center_context = []
                for sub_item in item[key]:
                    temp = DataFrame.as_matrix(co_mat[jj].loc[sub_item, :])
                    temp[np.isnan(temp)] = 0
                    temp = list(temp)
                    center_context.append(temp)
                temp_data.append([center_word, center_context])
            jj = jj + 1
            training_data.append(temp_data)
        return training_data

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

    def feed_forward(self, input_to_hidden_weights, hidden_to_output_weights, center_word):
        output_vector1 = np.dot(input_to_hidden_weights.T, center_word)
        output_vector2 = np.dot(hidden_to_output_weights.T, output_vector1)
        y_pred = self.activation_function(output_vector2)
        return y_pred, output_vector1, output_vector2

    def back_propagation(self, error, output_vector1, hidden_to_output_weights,
                         input_to_hidden_weights, center_word):
        der_w2 = np.outer(output_vector1, error)
        der_w1 = np.outer(center_word, np.dot(hidden_to_output_weights, error))

        self.input_to_hidden_weights = input_to_hidden_weights - (self.setting['learning_rate'] * der_w1)
        self.hidden_to_output_weights = hidden_to_output_weights - (self.setting['learning_rate'] * der_w2)

    def train_network(self, training_data, decomposed_context):

        unique_terms = self.word_count_per_document(decomposed_context)
        dim = np.power(self.setting['window_size'], 2) + 1

        vector_holder = []

        for c in range(len(unique_terms)):
            self.input_to_hidden_weights = np.random.uniform(-0.8, 0.8, (len(unique_terms[c]), dim))
            self.hidden_to_output_weights = np.random.uniform(-0.8, 0.8, (dim, len(unique_terms[c])))

            for i in range(0, self.setting['epoch']):
                self.loss = 0
                for center_word, center_context in training_data[c]:
                    y_pred, output_vector1, output_vector2 = self.feed_forward(self.input_to_hidden_weights,
                                                                               self.hidden_to_output_weights,
                                                                               center_word)

                    error = np.sum([np.subtract(y_pred, word) for word in center_context],
                                   axis=0)

                    self.back_propagation(error, output_vector1, self.hidden_to_output_weights,
                                          self.input_to_hidden_weights, center_word)

                    self.loss += -np.sum([output_vector2[word.index(1)] for word in center_context]) + len(
                        center_context) * np.log(np.sum(np.exp(output_vector2)))

                print('EPOCH:', i, 'LOSS:', self.loss)
                vector_holder.append(self.input_to_hidden_weights)
        return vector_holder


def main():
    ann_obj = SimpleNeuralNetwork()

    main_corpora_address = 'datasets\\raw_ner_data.csv'
    test_corpora_address = 'datasets\\input_text_for_test.txt'

    main_corpora = pd.read_csv(main_corpora_address)
    test_corpora = dh_obj.load_txt_data_as_list(test_corpora_address)

    # test_corpora = tk_obj.sent_tokenizer(test_corpora)
    test_corpora = tk_obj.word_tokenizer(test_corpora)

    # decomposed_context, complete_context = tk_obj.document_extraction(test_corpora, test=True)
    co_mat, co_occ_dictionary, co_occ_dictionary_unique = co_occ_obj.build_co_occurrence_matrix(test_corpora)
    training_data = ann_obj.generate_training_data(co_mat, co_occ_dictionary_unique)
    vector = ann_obj.train_network(training_data, test_corpora)
    print(vector)


if __name__ == '__main__':
    main()
