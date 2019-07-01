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
        if len(data) > 0:
            data = data[0]
        else:
            pass
        pattern = re.compile(r'([!\.\،\!\;\?⸮؟]+)[ \n]+')
        temp = pattern.sub(r'\1\n\n', data)
        return [sentence.replace('\n', ' ').strip() for sentence in temp.split('\n\n') if sentence.strip()]

    def document_extraction(self, data, test=False):
        if isinstance(data, list) is False and test is True:
            raise TypeError("Please assign test parameter to False")
        elif isinstance(data, list) is True and test is False:
            raise TypeError("Please assign test parameter to True")
        else:
            pass

        if len(data) == 1:
            pivot = data[0]
            pass
        if test and len(data) > 1:
            testdata = list(x for x in data)
            complete_context = testdata
            pivot = testdata
        if not test:
            corpus = data['words']
            complete_context = list(y for y in corpus)
            pivot = corpus
        i = 0
        doc_holder = []
        temp_holder = []
        for word in pivot:
            if word == '،':
                word = word.strip('،')
                doc_holder.insert(i, temp_holder)
                i = i + 1
                temp_holder = []
                pass
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
        doc_holder.append(temp_holder)
        return doc_holder, complete_context

    # before start tokenizing words, first you need to run the sentence tokenizer function
    def word_tokenizer(self, data):
        pattern = re.compile(r'([؟!\?]+|\d[\d\.:/\\]+|[:\.،؛»\]\)\}"«\[\(\{])')
        # text = pattern.sub(r' \1 ', data.replace('\n', ' ').replace('\t', ' '))
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
