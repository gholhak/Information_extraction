from data_utils import DataHandler
from FarsiTokenizer import Tokenizer
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import nltk

from sklearn.feature_extraction import DictVectorizer

filename = 'data\\train_fold1.txt'
load_file_name = 'data\\my_ner.csv'
tk_obj = Tokenizer()
dh_obj = DataHandler()
v = DictVectorizer(sparse=False)


def main():
    text = "روحانی به نیویورک رفت."

    # data = dh_obj.load_farsi_tokens(filename)
    # dh_obj.extract_farsi_tokens(data)
    # data = pd.read_csv('data\\my_ner.csv')
    # data = dh_obj.loadCSV(load_file_name)
    # data = data.fillna(method="ffill")
    _sents = tk_obj.sent_tokenizer(text)
    _words = tk_obj.word_tokenizer(_sents)
    _big = nltk.bigrams(_words)


if __name__ == '__main__':
    main()
