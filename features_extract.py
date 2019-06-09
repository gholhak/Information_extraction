from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from classifier import MemoryTagger
import numpy as np
from pandas import DataFrame




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
