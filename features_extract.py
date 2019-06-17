import math
import numpy as np
from pandas import DataFrame


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

    def computeIDFDict(self, countDict, whole_doc):
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


class Co_occurrence:
    def __init__(self):
        pass

    def extract_unique_terms(self, doc, corpus):
        class_holder = {}
        unique_terms_dictionary = []
        i = 0
        for word in doc:
            if word not in unique_terms_dictionary:
                unique_terms_dictionary.append(word)
                class_holder[word] = corpus.iloc[i, 1:21]
                i = i + 1

        return unique_terms_dictionary, class_holder

    def compute_count(self, unique_terms, doc):
        doctf = []
        uniqueDict = {}
        d = np.zeros((len(doc), len(unique_terms)))
        aa = DataFrame(data=d, dtype=int, columns=unique_terms)
        # if you want to get matrix as a numpy ndarray
        # co_mat = np.zeros((len(doc), len(unique_terms)), dtype=int)
        j = 0
        for uni_words in unique_terms:
            i = 0
            for _doc in doc:
                for doc_word in _doc:
                    if uni_words == doc_word:
                        aa.iloc[i, j] += 1
                i = i + 1
            j = j + 1
        return aa
