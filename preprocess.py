
import json
import requests
from elasticsearch import Elasticsearch
import glob
import os
import nltk
import re
import pprint
from nltk import word_tokenize, sent_tokenize


class DataHandler:
    def __init__(self, host, port, datadirectory):
        self.host = host
        self.port = port
        self.dataDirectory = datadirectory

    def server_connection(self):
        es = Elasticsearch([{'host': self.host, 'port': self.port}])
        return es

    def query_designer(self):
        doc = {
            'size': 1000,
            'query': {
                'match_all': {}
            }
        }
        return doc

    def elastic_server_extraction(self, doc, es):
        res = es.search(index='tripadvisor', doc_type='tripadvisor', body=doc)
        data = []
        with open('mydata.json', 'w', encoding='utf-8') as outfile:
            for hit in res['hits']['hits']:
                myhit = json.dumps(hit)
                finalHit = json.loads(myhit)
                data.append(finalHit['_source'])
                outfile.write(json.dumps(
                    finalHit['_source'], indent=10, ensure_ascii=False,))
        return data

    def save_as_text(self, data):
        with open('out.txt', 'w') as f:
            f.writelines(json.dumps(data))

    def load_as_list(self):
        with open('out.txt', 'r') as f:
            new_raw_data = json.loads(f.read())
        return new_raw_data