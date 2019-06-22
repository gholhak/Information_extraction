from concept_extraction.features_extract import TF_IDF
from concept_extraction.FarsiTokenizer import Tokenizer
from data_handler.data_utils import DataHandler
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
the classes are instantiated here.
'''
df = DataFrame()
tf_obj = TF_IDF()
tk_obj = Tokenizer()
dh_handler = DataHandler()

'''
The NER classification dataset path
'''
corpus_address = 'datasets\\raw_ner_data.csv'


def main():
    # load the ner dataset as csv file using pandas framework
    corpus = pd.read_csv(corpus_address)
    # fill the 'na' values in the dataset using ffill method
    corpus = corpus.fillna(method='ffill')

    '''
    The variables to store tf-idf operation outputs are defined here.
    tfdict_per_document = stores tf computation values
    unique_terms_holder = stores uniques terms in each document
    idfDict_holder = stores idf values for each term in respective document
    tf_idf_holder = stores the tf-idf scores for each term in each document
    '''
    tf_idf_holder = []

    '''
    This function, extracts documents from the NER csv dataset.
    the method is so naive. So, do not hesitate to code your own function
    '''
    doc_holder = tk_obj.ner_data_document_extraction(corpus)

    '''
    Iterates over each document in doc_holder list
    '''

    tf_holder = tf_obj.computeTF(doc_holder)
    unique_terms_holder = tf_obj.number_of_documents_containing_terms(tf_holder)
    idfDict = tf_obj.compute_idf_dict(unique_terms_holder, tf_holder)

    for i in range(len(tf_holder)):
        tfidfvec = tf_obj.computeCorpusTFIDFDict(tf_holder[i], idfDict[i])
        tf_idf_holder.append(tfidfvec)

    print(tf_idf_holder)


if __name__ == '__main__':
    main()
