from data_handler.data_utils import DataHandler
from concept_extraction.FarsiTokenizer import Tokenizer
from concept_extraction.features_extract import CountVector
from concept_extraction.features_extract import CoOccurrence
import itertools
import pandas as pd
from collections import Counter

setting = {}
setting['window_size'] = 1
data_address = 'datasets\\raw_ner_data.csv'

dh_obj = DataHandler()
tk_obj = Tokenizer()
co_occurrence_obj = CoOccurrence(setting)


def main():
    test_text1 = [['می کند', 'زندگی', 'روحانی', 'چندین', 'نیویورک', 'در'],
                  ['نیویورک', 'سفر', 'کرد', 'سفر', 'نیویورک', 'به', 'روحانی']]

    test_text2 = [['روحانی', 'به', 'نیویورک', 'سفر', 'کرد', 'در', 'نیویورک', 'چندین', 'روحانی', 'زندگی', 'می کند']]

    # to see the full matrix, please put a breakpoint at the next line.
    # Observce the results in debuger console.
    co_mat = co_occurrence_obj.build_co_occurrence_matrix(test_text2)
    print(co_mat)


if __name__ == '__main__':
    main()
