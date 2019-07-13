import math
import numpy as np
from pandas import DataFrame
from collections import Counter
from data_handler.data_utils import DataHandler
import pandas as pd

dh_obj = DataHandler()
csv_columns = ['words', 'PERSON', 'NORP', 'FACILITY', 'ORGANIZATION', 'GPE', 'LOCATION', 'PRODUCT', 'EVENT',
               'WORK_OF_ART', 'LAW',
               'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'MEASUREMENT', 'ORDINAL', 'CARDINAL', 'MISC',
               'PUNC', 'O']


class TF_IDF:
    def __init__(self):
        pass

    def computeTF(self, data):
        doctf = []
        for _doc in data:
            reviewTFDict = {}
            for words in _doc:
                if words in reviewTFDict:
                    reviewTFDict[words] += 1
                else:
                    reviewTFDict[words] = 1
            for word in reviewTFDict:
                reviewTFDict[word] = reviewTFDict[word] / len(data)
            doctf.append(reviewTFDict)
        return doctf

    def number_of_documents_containing_terms(self, tfDict):
        countHolder = []
        countDictt = {}
        for i in range(len(tfDict)):
            pivot = tfDict[i]
            for j in range(len(tfDict)):
                countDict = tfDict[j]
                for word in pivot:
                    if word in countDict:
                        countDictt[word] += 1
        countHolder.append(countDictt)
        return countHolder

    def compute_idf_dict(self, countDict, whole_doc):
        docsIDF = []
        for _doc in countDict:
            idfDict = {}
            for word in _doc:
                idfDict[word] = math.log(len(whole_doc) / _doc[word])
            docsIDF.append(idfDict)
        return docsIDF

    def computeCorpusTFIDFDict(self, reviewTFDict, idfDict):
        reviewTFIDFDict = {}
        # For each word in the review, we multiply its tf and its idf.
        for word in reviewTFDict:
            reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
        return reviewTFIDFDict


class CountVector:
    def __init__(self):
        pass

    def extract_unique_terms(self, corpus):
        class_holder = {}
        unique_terms_dictionary = []
        value_holder = []
        i = 0
        for word in corpus['words']:
            if word not in unique_terms_dictionary:
                unique_terms_dictionary.append(word)
                # value_holder.append(corpus.iloc[i, 0:21])
                class_holder[word] = corpus.iloc[i, 1:22]
                i = i + 1
        # for i in range(len(corpus)):
        #     if corpus.iloc[i, :] not in value_holder:
        #         value_holder.append(corpus.iloc[i, :])
        return unique_terms_dictionary, class_holder

    def compute_count(self, unique_terms, doc):
        doctf = []
        uniqueDict = {}
        d = np.zeros((len(doc), len(unique_terms)))
        aa = DataFrame(data=d, dtype=int, columns=unique_terms)
        # if you want to get matrix as a numpy ndarray
        co_mat = np.zeros((len(doc), len(unique_terms)), dtype=int)
        j = 0
        for uni_words in unique_terms:
            i = 0
            for _doc in doc:
                for doc_word in _doc:
                    if uni_words == doc_word:
                        # aa.iloc[:, j] += 1
                        co_mat[i, j] += 1
                i = i + 1
            j = j + 1
        return co_mat


class CoOccurrence:
    def __init__(self, setting):
        self.window_size = setting['w_size']

    def get_vector(self, key):
        # co_mat = self.binerize_co_occurrence(self.mat_holder)
        co_mat = self.mat_holder
        center_word = DataFrame.as_matrix(co_mat[0].loc[key, :])
        center_word[np.isnan(center_word)] = 0
        center_word = np.divide(center_word, np.sum(center_word))
        return list(center_word)

    def binerize_co_occurrence(self, co_mat):
        for i in range(0, len(co_mat[0])):
            for j in range(0, len(co_mat[0])):
                if np.isnan(co_mat[0].iloc[i, j]):
                    co_mat[0].iloc[i, j] = 0
                else:
                    co_mat[0].iloc[i, j] = 1
        return co_mat

    def extract_unique_terms(self, data):
        uni = []
        for term in data:
            if term not in uni:
                uni.append(term)
        return uni

    def padding(self, array, vector_indicator):
        pad = ([])
        difference = vector_indicator - np.size(array, 0)
        for i in range(difference):
            pad = np.append(pad, 0)
        array = np.append(array, pad)
        return array

    def build_vector_from_co_mat(self, co_mat, co_occurrence_dictionary):
        terms_vector_holder = []
        jj = 0
        for item in co_occurrence_dictionary:
            for sub_item in item:
                result_array = ([])
                for key in sub_item:
                    result = DataFrame.as_matrix(co_mat[jj].loc[key, :])
                    result[np.isnan(result)] = 0
                    result_array = np.append(result_array, result)
                result_array[np.isnan(result_array)] = 0
                vector_indicator = (np.power(self.window_size, 2) + 1) * len(co_mat[jj])
                if len(result_array) < vector_indicator:
                    full_vector = self.padding(result_array, vector_indicator)
                    terms_vector_holder.append(full_vector)
                else:
                    full_vector = self.padding(result_array, vector_indicator)
                    terms_vector_holder.append(full_vector)
            jj = jj + 1
        return terms_vector_holder

    def build_co_occurrence_matrix(self, doc):
        if any(isinstance(i, list) for i in doc) is False:
            corpus = [list(x for x in doc)]
        else:
            corpus = doc
        self.mat_holder = []
        dict_holder = []
        co_occ_unique_holder = []
        unique_labels = []
        binary_co_occurrence = []
        for sen in corpus:
            co_occ_unique = {ii: Counter({jj: 0 for jj in sen if jj != ii}) for ii in sen}
            co_occ = []
            i = 0
            for item in sen:
                co_occ.append([item])
                co_occ[i] = Counter(sen[i])
                i = i + 1
            for ii in range(len(sen)):
                # iterates from beginning of the document to reach the window_size max value
                if ii < self.window_size:
                    c = Counter(sen[0:ii + self.window_size + 1])
                    co_occ[ii].clear()
                    co_occ[ii] = co_occ[ii] + c
                    del c[sen[ii]]
                    co_occ_unique[sen[ii]] = co_occ_unique[sen[ii]] + c

                # keep distance from the last elements with regard to window_size
                elif ii > len(sen) - (self.window_size + 1):
                    c = Counter(sen[ii - self.window_size::])
                    co_occ[ii].clear()
                    co_occ[ii] = co_occ[ii] + c
                    del c[sen[ii]]
                    co_occ_unique[sen[ii]] = co_occ_unique[sen[ii]] + c
                # capture terms from the center of a document
                else:
                    c = Counter(sen[ii - self.window_size:ii + self.window_size + 1])
                    co_occ[ii].clear()
                    co_occ[ii] = co_occ[ii] + c
                    del c[sen[ii]]
                    co_occ_unique[sen[ii]] = co_occ_unique[sen[ii]] + c

            uni = self.extract_unique_terms(sen)
            unique_labels.append(uni)
            co_occ_mat_for_docs = pd.DataFrame(co_occ_unique, columns=uni, index=uni)
            self.mat_holder.append(co_occ_mat_for_docs)
            dict_holder.append(co_occ)
            co_occ_unique_holder.append(co_occ_unique)
        return self.mat_holder, dict_holder, co_occ_unique_holder, unique_labels
