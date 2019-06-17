from features_extract import Co_occurrence, TF_IDF
from FarsiTokenizer import Tokenizer
from data_utils import DataHandler
from pandas import DataFrame
import pandas as pd

df = DataFrame()
tk_obj = Tokenizer()
co_obj = Co_occurrence()
tf_obj = TF_IDF()
dh_handler = DataHandler()


def main():
    corpus_file_address = 'data\\sample.csv'
    # load the ner dataset as csv file using pandas framework
    raw_corpus = pd.read_csv(corpus_file_address)
    # fill the 'na' values in the dataset using ffill method
    raw_corpus = raw_corpus.fillna(method='ffill')

    # convert the corpus to list of strings
    corpus_as_a_list = list(x for x in raw_corpus['words'])

    # convert membership matrix of the tags to a single column with multiple class
    tags_list = dh_handler.mem_to_single_column_classification()

    # consider each sentence in corpus as a document
    separated_documents = tk_obj.ner_data_document_extraction(raw_corpus)
    # extract unique tokens from the corpus
    all_unique_terms, correspoding_class_values = co_obj.extract_unique_terms(corpus_as_a_list, raw_corpus)
    # count the co-occurance of each term in for each document
    co_occurence_matrix = co_obj.compute_count(all_unique_terms, separated_documents)
    t_mat = co_occurence_matrix.T
    
    # mydata = pd.DataFrame(co_occurence_matrix, columns=all_unique_terms)


if __name__ == '__main__':
    main()
