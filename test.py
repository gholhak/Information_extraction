from data_utils import DataHandler
from FarsiTokenizer import Tokenizer
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn.ensemble import RandomForestClassifier
import numpy as np

filename = 'data\\train_fold1.txt'
load_file_name = 'data\\my_ner.csv'
tk_obj = Tokenizer()
dh_obj = DataHandler()


def feature_map(word):
    return np.ndarray([word.istitle(), word.islower])


def main():
    # data = dh_obj.load_farsi_tokens(filename)
    # dh_obj.extract_farsi_tokens(data)
    data = pd.read_csv('data\\my_ner.csv')
    data = data.fillna(method="ffill")
    print(data.tail(10))


if __name__ == '__main__':
    main()
