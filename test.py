from data_utils import DataHandler
from FarsiTokenizer import Tokenizer
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction import DictVectorizer
from classifier import MemoryTagger
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from features_extract import FeatureTransformer
from sklearn.ensemble import RandomForestClassifier

filename = 'data\\train_fold1.txt'
load_file_name = 'data\\my_ner.csv'

tk_obj = Tokenizer()
dh_obj = DataHandler()
mt_obj = MemoryTagger()
v = DictVectorizer(sparse=False)
clf = SVC(kernel='linear', C=1)
ft = FeatureTransformer()


def main():
    # data = dh_obj.load_farsi_tokens(filename)
    # dh_obj.extract_farsi_tokens(data)
    _data = pd.read_csv('data\\my_ner.csv')
    _data = _data.fillna(method="ffill")

    train_data = _data.iloc[0:9]

    mt_obj.fit(train_data['word'], train_data['tag'])

    print(train_data['word'])

    # ft.fit(train_data, train_data['tag'])
    # enc = ft.transform(train_data)

    pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()),
                                       ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),
                             X=train_data, y=train_data['tag'], cv=5)

    report = classification_report(y_pred=pred, y_true=train_data['tag'])
    print(report)


# words = _data["word"].values.tolist()
# tags = _data["tag"].values.tolist()
#
# scores = cross_val_score(clf, words, tags, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
