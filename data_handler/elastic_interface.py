import json
from elasticsearch import Elasticsearch


class Extract_data:
    def __init__(self, HOST, PORT, data_dir):
        self.host = HOST
        self.port = PORT
        self.dir = data_dir

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
        res = es.search(index='instatrip', body=doc)
        data = []
        with open('FarsiData.json', 'w', encoding='utf-8') as outfile:
            for hit in res['hits']['hits']:
                myhit = json.dumps(hit)
                finalHit = json.loads(myhit)
                data.append(finalHit['_source'])
                outfile.write(json.dumps(
                    finalHit['_source'], indent=10, ensure_ascii=False, ))
        return data

    def save_as_text(self, data):
        with open('FarsiData.txt', 'w', encoding='UTF-8') as f:
            f.writelines(json.dumps(data))

    def load_as_list(self):
        with open('FarsiData.txt', 'r', encoding='UTF-8') as f:
            new_raw_data = json.loads(f.read())
        return new_raw_data
