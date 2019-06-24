from concept_extraction.features_extract import CountVector, TF_IDF
from concept_extraction.FarsiTokenizer import Tokenizer
from data_handler.data_utils import DataHandler
from pandas import DataFrame
import pandas as pd

df = DataFrame()
tk_obj = Tokenizer()
co_obj = CountVector()
tf_obj = TF_IDF()
dh_handler = DataHandler()


def main():
    corpus_file_address = 'datasets\\raw_ner_data.csv'
    # load the ner dataset as csv file using pandas framework
    raw_corpus = pd.read_csv(corpus_file_address)
    # fill the 'na' values in the dataset using ffill method
    raw_corpus = raw_corpus.fillna(method='ffill')

    # convert the corpus to list of strings
    corpus_as_a_list = list(x for x in raw_corpus['words'])

    # convert membership matrix of the tags to a single column with multiple class
    # tags_list = dh_handler.mem_to_single_column_classification()

    # consider each sentence in corpus as a document
    separated_documents = tk_obj.ner_data_document_extraction(raw_corpus)
    # extract unique tokens from the corpus
    all_unique_terms, unique_terms_with_labels = co_obj.extract_unique_terms(raw_corpus)
    # count the co-occurance of each term in for each document
    count_matrix = co_obj.compute_count(all_unique_terms, separated_documents)
    count_matrix = pd.DataFrame(count_matrix, columns=all_unique_terms)
    t_mat = count_matrix.T

    # output final dataset
    dh_handler.merg(unique_terms_with_labels, t_mat)


if __name__ == '__main__':
    main()
