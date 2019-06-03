from preprocess import DataHandler
from IR import InformationExtraxtion
import langdetect
from FarsiTokenizer import Tokenizer
# from Chunker import FindChunks
# from POSTaggerr import POSTagger
import nltk
import json
import re

'''
Constants are defined here!
'''
HOST = "89.219.110.142"
PORT = 5500
DATA_DIRECTORY = "E:\\projects\\Samira\\mydatanew.json"

'''
The main function of the program.
Please note that the initialization of the objects are inside the main function.
'''
ir_obj = InformationExtraxtion()

dh_obj = DataHandler(HOST, PORT, DATA_DIRECTORY)
tk_obj = Tokenizer()

filename = "data\\train_fold1.txt"


def main():
    ner_corpora = []
    fuzzy_ner = []
    dh_obj = DataHandler(HOST, PORT, DATA_DIRECTORY)
    # conn = dh_obj.server_connection()
    # qry = dh_obj.query_designer()
    # raw_data = dh_obj.elastic_server_extraction(qry, conn)

    raw_data = "علی به تهران سفر کرد. سفر اون بخاطر دیدار با پدر خود بود."
    # langdetect.detect()
    # for entry in raw_data:
    #     selected_data = entry['comments']
    #     for rows in selected_data:
    #         flag = langdetect.detect(rows)
    #         if flag == "fa":
    sents = tk_obj.sent_tokenizer(raw_data)
    words = tk_obj.word_tokenizer(sents)
    train_fold = tk_obj.load_as_list_Farsi(filename)
    for _word in words:
        for _w in _word:
            _top_match = ir_obj.fuzzzy(train_fold, _w)
            fuzzy_ner.append(_top_match)
            # _w = _w.strip('.')
            # parsed = tk_obj.regexp_parser(train_fold, _w, 'pos')
            # ner_corpora.append(parsed)

    print('hi')


'''
The entry point for the program
'''
if __name__ == '__main__':
    main()
