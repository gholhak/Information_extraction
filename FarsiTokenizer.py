try:
    from win_unicode_console import enable

    enable()
except ImportError:
    pass

#
import re
from collections import OrderedDict
import math

vocab = []


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class Tokenizer():
    def __init__(self, *args, **kwargs):
        pass

    def dot_product(self, vector_x, vector_y):
        dot = 0.0
        for e_x, e_y in zip(vector_x, vector_y):
            dot += e_x * e_y
        return dot

    def magnitude(vector):
        mag = 0.0
        for index in vector:
            mag += math.pow(index, 2)
        return math.sqrt(mag)

    def sent_tokenizer(self, data):
        pattern = re.compile(r'([!\.\،\!\;\?⸮؟]+)[ \n]+')
        temp = pattern.sub(r'\1\n\n', data)
        return [sentence.replace('\n', ' ').strip() for sentence in temp.split('\n\n') if sentence.strip()]

    def ner_data_document_extraction(self, data):
        # corpus = ' '.join(corpus)
        corpus = list(x for x in data['words'])
        doc_holder = []
        temp_holder = []
        i = 0
        for word in corpus:
            if word == '،':
                word = word.strip('،')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '!':
                word = word.strip('!')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '.':
                word = word.strip('.')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == ';':
                word = word.strip(';')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '؟':
                word = word.strip('؟')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '…':
                word = word.strip('…')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '(':
                word = word.strip('(')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == ')':
                word = word.strip(')')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '...':
                word = word.strip('...')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '?':
                word = word.strip('?')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            if word == '\u200c':
                word = word.strip('\u200c')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
            else:
                temp_holder.append(word)
        return doc_holder

    # before start tokenizing words, first you need to run the sentence tokenizer function
    def word_tokenizer(self, data):
        pattern = re.compile(r'([؟!\?]+|\d[\d\.:/\\]+|[:\.،؛»\]\)\}"«\[\(\{])')
        text = pattern.sub(r' \1 ', data.replace('\n', ' ').replace('\t', ' '))
        sent_holder_for_each_word = []
        i = 0
        for each_sent in data:
            word_holder = []
            doc_string = str(each_sent)
            list_of_tokens = doc_string.strip().split()
            j = 0
            for word in list_of_tokens:
                if len(word.strip("\u200c")) != 0:
                    word_holder.insert(j, word)
                    j = j + 1
            list_of_tokens.clear()
            sent_holder_for_each_word.insert(i, word_holder)
            i = i + 1
        return sent_holder_for_each_word
