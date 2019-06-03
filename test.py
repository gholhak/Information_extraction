
import json
import re
import codecs
from collections import Counter
import itertools
import re
import json

from FarsiTokenizer import Tokenizer
myobj = Tokenizer()

text_for_test = 'data/test.txt'
farsi_data = 'data/FarsiData.txt'
TAGG_data = 'data/UPC-2016.txt'


def main():
    _pos = {}
    text = myobj.load_as_list_Farsi(text_for_test)
    tagg = myobj.load_as_list_Farsi(TAGG_data)
    sents = myobj.sent_tokenizer(str(text))
    words = myobj.word_tokenizer(sents)
    fa = myobj.load_as_list_Farsi(farsi_data)
    _pos_data = myobj.regexp_parser(tagg)


if __name__ == '__main__':
    main()
