from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer
from concept_extraction.features_extract import CountVector
from concept_extraction.features_extract import CoOccurrence
import pandas as pd

dh_obj = DataHandler()
tk_obj = Tokenizer()
count_vector_obj = CountVector()
co_occurrence_obj = CoOccurrence()
data_address = 'datasets\\ner.txt.csv'


def main():
    raw_corpra = pd.read_csv(data_address)
    raw_corpra_as_list = tk_obj.ner_data_document_extraction(raw_corpra)

    for doc in raw_corpra_as_list:
        unique_doc = count_vector_obj.extract_unique_terms(raw_corpra_as_list)
        co_mat = co_occurrence_obj.build_co_occurrence_matrix(doc)


if __name__ == '__main__':
    main()
