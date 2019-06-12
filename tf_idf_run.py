
from features_extract import TF_IDF
from FarsiTokenizer import Tokenizer
from data_utils import DataHandler
import pandas as pd
from pandas import DataFrame

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
doc = 'data\\ner.txt.csv'

def main():
    # load the ner dataset as csv file using pandas framework
    corpus = pd.read_csv(doc)
    # fill the 'na' values in the dataset using ffill method
    corpus = corpus.fillna(method='ffill')

    '''
    The variables to store tf-idf operation outputs are defined here.
    tfdict_per_document = stores tf computation values
    unique_terms_holder = stores uniques terms in each document
    idfDict_holder = stores idf values for each term in respective document
    tf_idf_holder = stores the tf-idf scores for each term in each document
    '''
    tfdict_per_document = []
    unique_terms_holder = []
    idfDict_holder = []
    tf_idf_holder = []

    '''
    This function, extracts documents from the NER csv dataset.
    the method is so naive. So, do not hesitate to code your own function
    '''
    doc_holder = tk_obj.ner_data_document_extraction(corpus)

    '''
    Iterates over each document in doc_holder list
    '''
    for document in doc_holder:
        # computes term frequency of each word in each document
        tf_holder = tf_obj.computeTermsTF(document)
        # append each document to document holder list
        tfdict_per_document.append(tf_holder)
        num_doc = tf_obj.number_of_documents_containing_terms(tfdict_per_document)
        unique_terms_holder.append(num_doc)

    for uni_terms in unique_terms_holder:
        idfDict = tf_obj.computeIDFDict(uni_terms, tfdict_per_document)
        idfDict_holder.append(idfDict)

    for i in range(len(tfdict_per_document)):
        tfidfvec = tf_obj.computeCorpusTFIDFDict(tfdict_per_document[i], idfDict_holder[i])
        tf_idf_holder.append(tfidfvec)

    print(tf_idf_holder)


if __name__ == '__main__':
    # please execute the program in debug mode 
    main()
