from data_handler.data_utils import DataHandler

import langdetect
from concept_extraction.FarsiTokenizer import Tokenizer
import nltk
import json

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

def main():
    _corpora = []
    dh_obj = DataHandler(HOST, PORT, DATA_DIRECTORY)
    conn = dh_obj.server_connection()
    qry = dh_obj.query_designer()
    raw_data = dh_obj.elastic_server_extraction(qry, conn)

    # langdetect.detect()
    for entry in raw_data:
        sents = tk_obj.sent_tokenizer(str(raw_data))
    words = tk_obj.word_tokenizer(sents)

    rel_corpora = ir_obj.relation_extraction(_corpora)
    print(rel_corpora)


'''
The entry point for the program
'''
if __name__ == '__main__':
    main()
