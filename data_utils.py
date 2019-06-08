import json
import re
import codecs
import csv


class DataHandler:
    def __init__(self):
        pass

    def save_as_text(self, data):
        with open('FarsiData.txt', 'w', encoding='UTF-8') as f:
            f.writelines(json.dumps(data))

    def load_as_list(self):
        with open('FarsiData.txt', 'r', encoding='UTF-8') as f:
            new_raw_data = json.loads(f.read())
        return new_raw_data

    def load_farsi_tokens(self, filename):
        vocab = {}
        with codecs.open(filename, encoding='UTF-8') as f:
            lines = f.readlines()
            # for line in f:
            #     vocab.update(line.strip().split())
        return lines

    def extract_farsi_tokens(self, source):
        data_obj = []
        for t in source:
            # NER parser
            t = t.strip('\n')
            full_match_group = re.findall('(\S+)', t)
            # key = re.findall(r'\w\s(.*)', t)
            if full_match_group:
                key = full_match_group[0]
                val = full_match_group[1]
                _data = [key, val]
                data_obj.append(_data)

        with codecs.open('data\\my_ner.csv', mode='w', encoding='UTF-8') as csv_file:
            for rows in data_obj:
                wr = csv.writer(csv_file)
                wr.writerow(rows)

    def loadCSV(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as csvreader:
            myreader = csv.reader(csvreader)
            mydata = list(myreader)
        return mydata
