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
#     conn = dh_obj.server_connection()
#     qry = dh_obj.query_designer()
#     raw_data = dh_obj.elastic_server_extraction(qry, conn)

    raw_data = dh_obj.load_as_list() 

    for i in range(len(raw_data)):
        labeled = ir_obj.tokenizer(raw_data[i]['body'])
        _corpora.append(labeled)


'''
The entry point for the program
'''
if __name__ == '__main__':
    main()
