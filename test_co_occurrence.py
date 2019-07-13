from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer
from concept_extraction.features_extract import CoOccurrence
import pandas as pd
import numpy as np

setting = {}
# determines the window-size
setting['w_size'] = 2
# the file address of the main corpora gathered by Fred
main_corpora_address = 'datasets\\raw_ner_data.csv'

# initialization of the objects
dh_obj = DataHandler()
tk_obj = Tokenizer()
co_occurrence_obj = CoOccurrence(setting)

# load main corpora as a dataframe object
main_corpora = pd.read_csv(main_corpora_address)
# a test corpora for the better understanding of the module
test_corpora = ['روحانی', 'به', 'نیویورک', 'سفر', 'کرد', 'در', 'نیویورک', 'چندین', 'روحانی', 'زندگی', 'می کند']


# the entry point of the module
def main():
    # specifies the context where the algorithm searches
    decomposed_context, complete_context = tk_obj.document_extraction(test_corpora, test=True)
    # the output of the main algorithm. co_mat is a co-occurrence of the corpora as a dataframe structure
    # dict_mat is the co-occurrence as a dictionary
    co_mat_holder, dict_holder, co_occ_unique_holder, unique_labels = co_occurrence_obj.build_co_occurrence_matrix(
        complete_context)
    # dh_obj.save_data(co_mat_holder, unique_labels)
    vectors = co_occurrence_obj.build_vector_from_co_mat(dict_holder, co_mat_holder)
    print(vectors)

    # arr = np.zeros((len(vectors), len(vectors[0])))
    # for i in range(len(arr)):
    #     arr[i] = vectors[i]
    # np.savetxt('test.csv', arr, fmt="%d", delimiter=",")


if __name__ == '__main__':
    main()
