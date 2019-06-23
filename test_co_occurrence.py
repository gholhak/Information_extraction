from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer
from concept_extraction.features_extract import CoOccurrence
import pandas as pd

setting = {}
# determines the window-size
setting['window_size'] = 2
# the file address of the main corpra gathered by Fred
main_corpra_address = 'datasets\\raw_ner_data.csv'

# initialization of the objects
dh_obj = DataHandler()
tk_obj = Tokenizer()
co_occurrence_obj = CoOccurrence(setting)

# load main corpra as a dataframe object
main_corpra = pd.read_csv(main_corpra_address)
# a test corpra for the better understanding of the module
test_corpra = ['روحانی', 'به', 'نیویورک', 'سفر', 'کرد', 'در', 'نیویورک', 'چندین', 'روحانی', 'زندگی', 'می کند']


# the entry point of the module
def main():
    # specifies the context where the algorithm searches
    decomposed_context, complete_context = tk_obj.ner_data_document_extraction(test_corpra, test=True)
    # the output of the main algorithm. co_mat is a co-occurrence of the corpora as a dataframe structure
    co_mat = co_occurrence_obj.build_co_occurrence_matrix(decomposed_context)

    print(co_mat)


if __name__ == '__main__':
    main()
