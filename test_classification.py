import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from model_handler.model_util import model_handler

mh_obj = model_handler()
svm_clf = SVC(kernel='linear', C=1)


def main():
    _traindata = pd.read_csv('E:\\projects\\Samira\\datasets\\NER_data_single_column_tag.csv', sep=',')
    svm_clf.fit(_traindata, _traindata['tags'])
    mh_obj.save_model(svm_clf, path='model_svm')
    # m_svm = mh_obj.load_model(filename='m_svm')

    # If you want to perform cross-validation, uncomment this piece of code
    # pred = cross_val_predict(estimator=SVC(), X=_traindata, y=_traindata['tags'], cv=2)
    # report = classification_report(y_pred=pred, y_true=_traindata['tags'])


if __name__ == '__main__':
    main()
