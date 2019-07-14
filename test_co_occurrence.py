from concept_extraction.features_extract import CoOccurrence
from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer

"""
Setting is the configuration for the co occurrence class. 
W_size is the size of sliding window. Sliding window determines the
size of the context which the algorithm searches
"""
setting = {'w_size': 2}

"""
Initialization of the required class
"""
co_obj = CoOccurrence(setting)
dh_obj = DataHandler()
tk_obj = Tokenizer()

'''
The address of the input file. 
'''
main_corpora_address = "datasets\\input_text_for_test.txt"

test_corpora = dh_obj.load_txt_data_as_list(main_corpora_address)

'''
Tokenize the input corpora 
'''
test_corpora = tk_obj.word_tokenizer(test_corpora)

"""
IF YOU WANT TO SEE THE CO OCCURRENCE MATRIX AS A FANCY MATRIX,
PLEASE EXECUTE THE PROGRAM IN DEBUG MODE. THIS FUNCTION IS NOT DESIGNED FOR AN EXPLICIT USAGE.

NONETHELESS, I HAVE SAVED THE MATRIX AS A CSV FILE AND STORED IT IN THE FOLDER NAMELY 'output_files'

"""


def main():
    co_mat_holder, _, _, _ = co_obj.build_co_occurrence_matrix(test_corpora)
    co_mat_weighted = co_obj.weighted_co_occ(co_mat_holder)
    co_mat_weighted.to_csv('output_files\\co_occurrence_matrix.csv', sep=',', encoding='UTF-8')


if __name__ == '__main__':
    main()
