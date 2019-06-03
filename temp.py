from preprocess import DataHandler
import polyglot
import json
from FarsiTokenizer import Tokenizer
from langdetect import *
import parsivar
from preprocess import DataHandler
import codecs
from parsivar import *

# tk_obj = Tokenizer()
# filename = "FarsiData.json"
#
#
# def save_as_text(data):
#     with open('outfile.txt', 'w', encoding='UTF-8') as f:
#         f.writelines(data)
#
#
# def load_as_list(filename):
#     with codecs.open('data\FarsiData.txt', 'r', encoding='UTF-8') as f:
#     	new_raw_data = f.readlines()
#     return new_raw_data
#
#
# text = "دکتر روحانی به آمریکا سفر کرد. او به همراه بیست تن به این سفر بزرگ رفته است."
# print(detect(text))
# print(text)
#
# fa_data = load_as_list(filename)
# sents = tk_obj.sent_tokenizer(text)
# words = tk_obj.word_tokenizer(sents)
# save_as_text(sents)


myparser = DependencyParser()
my_tokenizer = Tokenizer()

sents = "به گزارش ایسنا سمینار شیمی آلی از امروز ۱۱ شهریور ۱۳۹۶ در دانشگاه علم و صنعت ایران آغاز به کار کرد. این سمینار تا ۱۳ شهریور ادامه می یابد"

sent_list = my_tokenizer.tokenize_sentences(sents)
parsed_sents = myparser.parse_sents(sent_list)

for depgraph in parsed_sents:
    print(depgraph)
