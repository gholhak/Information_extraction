import json
import re
import codecs
import csv
import numpy as np
import pandas as pd


class DataHandler:
    def __init__(self):
        pass

    def save_data(self, data, unique_labels):
        i = 0
        for i in range(len(data)):
            item = pd.DataFrame.as_matrix(data[i])
            item[np.isnan(item)] = 0

            item = pd.DataFrame(item, columns=unique_labels[i], index=unique_labels[i])
            address = 'models\\' + str(i) + '.csv'
            item.to_csv(address, encoding=None)

    def dict_to_csv(self, data):
        csv_columns = ['words', 'PERSON', 'NORP', 'FACILITY', 'ORGANIZATION', 'GPE', 'LOCATION', 'PRODUCT', 'EVENT',
                       'WORK_OF_ART', 'LAW',
                       'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'MEASUREMENT', 'ORDINAL', 'CARDINAL', 'MISC',
                       'PUNC', 'O']
        csv_file = "datasets\\NER_data_multiple_column_tag.csv"
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for i in range(len(data)):
                    writer.writerow(data.iloc[i, 0:22])
        except IOError:
            print("I/O error")

    @staticmethod
    def save_list_data_as_txt(data, filename):
        with open("datasets\\" + filename, 'w', encoding='UTF-8') as f:
            f.writelines(json.dumps(data))

    @staticmethod
    def load_txt_data_as_list(filename):
        with codecs.open(filename, encoding='UTF-8') as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def extract_tagged_tokens_as_csv(data, filename):
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

        with codecs.open('datasets\\' + filename, mode='w', encoding='UTF-8') as csv_file:
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
        data = np.genfromtxt('E:\\projects\\Samira\\datasets\\multiple_column_class.csv', dtype=int, delimiter=',')
        # target = datasets[:, 10:31]
        for i in range(1, len(data)):
            temp = np.nonzero(data[i, :])
            index_holder.append(temp[0][0])
        with open('E:\\projects\\Samira\\datasets\\single_column_class.csv', 'w') as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            writer.writerow(index_holder)

    def merg(self, unique_terms_with_labels, t_mat):
        with open('datasets\\NER_data_multiple_column_tag.csv', 'w') as file:
            values = []
            for key, val in unique_terms_with_labels.items():
                a = []
                for vec in t_mat.loc[key, :]:
                    a.append(vec)
                b = []
                for sub_val in val:
                    b.append(sub_val)
                joined_list = a + b
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                writer.writerow(joined_list)
