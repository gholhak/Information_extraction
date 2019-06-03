try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
    from win_unicode_console import enable

    enable()
except ImportError:
    pass

#
import re
import sys
from preprocess import DataHandler
import codecs
from collections import OrderedDict

vocab = []


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class Tokenizer():
    def __init__(self, *args, **kwargs):
        pass

    def sent_tokenizer(self, data):
        sent_holder = []
        pattern = re.compile(r'([!\.\?⸮؟]+)[ \n]+')
        temp = pattern.sub(r'\1\n\n', data)
        for sent in temp.split('\n\n'):
            sent = sent.strip('[]')
            sent = sent.strip("''")
            sent_holder.append(sent.replace('\n', '').strip())
        return sent_holder

    def word_tokenizer(self, data):
        sent_holder_for_each_word = []

        i = 0
        for each_sent in data:
            word_holder = []
            doc_string = str(each_sent)
            list_of_tokens = doc_string.strip().split()
            j = 0
            for word in list_of_tokens:
                if (len(word.strip("\u200c")) != 0):
                    word_holder.insert(j, word)
                    j = j + 1
            list_of_tokens.clear()
            sent_holder_for_each_word.insert(i, word_holder)
            i = i + 1
        return sent_holder_for_each_word

    def load_as_list_Farsi(self, filename):
        vocab = {}
        with codecs.open(filename, encoding='UTF-8') as f:
            lines = f.readlines()
            # for line in f:
            #     vocab.update(line.strip().split())
        return lines

    def regexp_parser(self, source, dest, varagin):
        res = []
        if varagin == "pos":
            for t in source:
                # NER parser
                t = t.strip('\n')
                full_match_group = re.findall('(\S+)', t)
                # key = re.findall(r'\w\s(.*)', t)
                if full_match_group:
                    key = full_match_group[0]
                    val = full_match_group[1]
                    if key == dest:
                        res.append([val, dest])
        if varagin == "ner":
            key = str(re.findall(r'\w+:?(?=\: )', t))
            val = str(re.findall(r'\:\s(.+)', t))
            key_temp = key.strip('[]')
            key = key_temp.strip("''")
            val_temp = val.strip('[]')
            val = val_temp.strip("''")

        # dic = DefaultListOrderedDict()
        # for i, k in enumerate(res):
        #     dic[k].append(res[i])
        #     print(dic)
        # for POStagger

        return res
