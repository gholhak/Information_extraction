import json
import re
import codecs
import csv
import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self):
        pass

    def save_list_data_as_txt(self, data, filename):
        with open("data\\" + filename, 'w', encoding='UTF-8') as f:
            f.writelines(json.dumps(data))

    def load_txt_data_as_list(self, filename):
        vocab = {}
        with codecs.open(filename, encoding='UTF-8') as f:
            lines = f.readlines()
            # for line in f:
            #     vocab.update(line.strip().split())
        return lines

    def extract_tagged_tokens_as_csv(self, data, filename):
        data_obj = []
        for t in data:
            # NER parser
            t = t.strip('\n')
            full_match_group = re.findall('(\S+)', t)
            # key = re.findall(r'\w\s(.*)', t)
            if full_match_group:
                key = full_match_group[0]
                val = full_match_group[1]
                _data = [key, val]
                data_obj.append(_data)

        with codecs.open('data\\' + filename, mode='w', encoding='UTF-8') as csv_file:
            for rows in data_obj:
                wr = csv.writer(csv_file)
                wr.writerow(rows)

    def load_csv_as_list(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as csvreader:
            myreader = csv.reader(csvreader)
            mydata = list(myreader)
        return mydata

    def mem_to_single_column_classification(self):
        index_holder = []
        data = np.genfromtxt('data\\ner.txt - Copy.csv', dtype=int, delimiter=',')
        print(len(data))
        for i in range(len(data)):
            if i != 0:
                for j in range(21):
                    print(data[i,:])
                    if data[i, j] != 0:
                        index_holder.append(data[0, j])
        return pd.DataFrame(index_holder)


        # with open('data\\tags.csv', 'w', newline='\n') as myfile:
        #     wr = csv.writer(myfile, dialect='excel')
        #     for row in index_holder:
        #         wr.writerow(row)
        #     myfile.close()
