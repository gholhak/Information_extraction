from data_utils import DataHandler
from FarsiTokenizer import Tokenizer
import pandas as pd
import pickle
from joblib import dump, load
from pandas import *
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction import DictVectorizer
from classifier import MemoryTagger
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from features_extract import FeatureTransformer
from sklearn.ensemble import RandomForestClassifier

filename = 'data\\train_fold1.txt'
load_file_name = 'data\\train_data.csv'

tk_obj = Tokenizer()
dh_obj = DataHandler()
mt_obj = MemoryTagger()
v = DictVectorizer(sparse=False)
clf = SVC(kernel='linear', C=1)
ft = FeatureTransformer()


def main():
    # train_data = pd.read_csv('data\\train_data.csv')
    # test_data = pd.read_csv('data\\test_fold.csv')
    # train_data = train_data.fillna(method="ffill")
    # test_data = test_data.fillna(method="ffill")

    # train_data = train_data.iloc[36200:36225]

    # tags = train_data["tag"].values.tolist()

    # mt_obj.fit(train_data['word'], tags)

    # ft.fit(train_data, train_data['tag'])
    # myword = ft.transform(train_data)
    # print(myword[:0])
    # myword = np.array(myword)

    # train_data['word'] = encoded_traindata
    # df = DataFrame(train_data)
    # df.to_csv('train_data.csv', sep=',', encoding='utf-8')
    # train_data = pd.read_csv('train_data.csv')
    _traindata = pd.read_csv('data\\trainingDataset.csv', sep=',')
    # pred = cross_val_predict(estimator=SVC(), X=_traindata, y=_traindata['tag'], cv=5)

    clf = RandomForestClassifier()
    clf.fit(_traindata, _traindata['tag'])

    saved_model = pickle.dumps(clf)
    svm_from_pickle = pickle.loads(saved_model)
    x_test = [[2, 4], [12, 1], [9, 1]]
    pred2 = svm_from_pickle.predict(x_test)
    print(pred2)
    # report = classification_report(y_pred=pred, y_true=_traindata['tag'])
    # print(report)


# words = _data["word"].values.tolist()
# tags = _data["tag"].values.tolist()
#
# scores = cross_val_score(clf, words, tags, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
