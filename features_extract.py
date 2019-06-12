from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from classifier import MemoryTagger
from pandas import DataFrame
import math
import numpy as np
from collections import defaultdict


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.memory_tagger = MemoryTagger()
        self.tag_encoder = LabelEncoder()

    def fit(self, X, y):
        words = X["word"].values.tolist()
        tags = X["tag"].values.tolist()
        self.memory_tagger.fit(words, tags)
        self.tag_encoder.fit(tags)
        return self

    def transform(self, X, y=None, ):
        words = X['word'].values.tolist()
        tags = X['tag'].values.tolist()
        out1 = []
        out2 = []
        for i in range(len(words)):
            w = words[i]
            t = tags[i]
            out1.append(self.tag_encoder.transform(self.memory_tagger.predict([w]))[0])
            out2.append(self.tag_encoder.transform(self.memory_tagger.predict([t]))[0])
        df = DataFrame(out1)
        df.to_csv('data\\trainingDataset.csv', sep=',', encoding='utf-8')
        return list(zip(out1, out2))


class TF_IDF:
    def __init__(self):
        pass

    def computeTermsTF(self, data):
        reviewTFDict = {}
        for words in data:
            if words in reviewTFDict:
                reviewTFDict[words] += 1
            else:
                reviewTFDict[words] = 1
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(data)
        return reviewTFDict

    def number_of_documents_containing_terms(self, tfDict):
        countDict = {}
        # Run through each review's tf dictionary and increment countDict's (word, doc) pair
        for review in tfDict:
            for word in review:
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1
        return countDict

    def computeIDFDict(self, countDict, whole_doc):
        idfDict = {}
        for word in countDict:
            idfDict[word] = math.log(len(whole_doc) / countDict[word])
        return idfDict

    def computeCorpusTFIDFDict(self, reviewTFDict, idfDict):
        reviewTFIDFDict = {}
        # For each word in the review, we multiply its tf and its idf.
        for word in reviewTFDict:
            reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
        return reviewTFIDFDict
