try:
    from win_unicode_console import enable

    enable()
except ImportError:
    pass

#
import re
import sys
from data_utils import DataHandler
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
