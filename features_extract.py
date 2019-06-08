from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from classifier import MemoryTagger
import numpy as np


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

    def transform(self, X, y=None):
        words = X['word'].values.tolist()
        out = []
        for i in range(len(words)):
            w = words[i]
            if i < len(words) - 1:
                wp = self.tag_encoder.transform(self.memory_tagger.predict([words[i + 1]]))[0]
            else:
                wp = self.tag_encoder.transform(['O'])[0]
            if i > 0:
                if words[i - 1] != ".":
                    wm = self.tag_encoder.transform(self.memory_tagger.predict([words[i - 1]]))[0]
                else:
                    wm = self.tag_encoder.transform(['O'])[0]
            else:
                wm = self.tag_encoder.transform(['O'])[0]
            out.append(self.tag_encoder.transform(self.memory_tagger.predict([w]))[0])
        return out
