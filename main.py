from preprocess import DataHandler
from IR import InformationExtraxtion
import nltk
import json
'''
Constants are defined here!
'''
HOST = "89.219.216.178"
PORT = 5500
DATA_DIRECTORY = "E:\\projects\\Samira\\mydatanew.json"

'''
The main function of the program.
Please note that the initialization of the objects are inside the main function.
'''
ir_obj = InformationExtraxtion()


def main():
    _corpora = []
    dh_obj = DataHandler(HOST, PORT, DATA_DIRECTORY)
    # conn = dh_obj.server_connection()
    # qry = dh_obj.query_designer()
    # raw_data = dh_obj.elastic_server_extraction(qry, conn)

    # dh_obj.save_as_text(raw_data)
    raw_data = dh_obj.load_as_list() 

    # for i in range(10): raw_data[i]['body']

    text = "I saw John when he was in U.s. We arranged a meetin and he talked about his fired Jessica. He told me that Jessica had been living in Paris for 10 years."
    labeled = ir_obj.tokenizer(text)
    _corpora.append(labeled)

    rel_corpora = ir_obj.relation_extraction(_corpora)
    print(rel_corpora)


'''
The entry point for the program
'''
if __name__ == '__main__':
    main()
