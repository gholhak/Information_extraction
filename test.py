
import json
import re
import codecs
from collections import Counter

from FarsiTokenizer import Tokenizer
tk_obj = Tokenizer()

vocab = []

text = "دکتر روحانی به نیویورک سفر کرد!"


def load_as_list():
    with codecs.open('data/UPC-2016.txt', encoding='UTF-8') as f:
        # new_raw_data = f.readlines()
        for line in f:
            vocab.append(line.strip().split())
    return vocab


_words = tk_obj.word_tokenizer(text)

data = load_as_list()

for words in _words:
    print(re.search(words, data))
