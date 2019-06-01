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
vocab = []


class Tokenizer():
    def __init__(self, *args, **kwargs):
        pass

    def sent_tokenizer(self, data):
        sent_holder = []
        pattern = re.compile(r'([!\.\?⸮؟]+)[ \n]+')
        temp = pattern.sub(r'\1\n\n', data)
        for sent in temp.split('\n\n'):
            sent_holder.append(sent.replace('\n', '').strip())
        return sent_holder

    def word_tokenizer(self, data):
        sent_holder_for_each_word = []
        word_holder = []
        i = 0
        for each_sent in data:
            doc_string = str(each_sent)
            list_of_tokens = doc_string.strip().split()
            j = 0
            for word in list_of_tokens:
                if(len(word.strip("\u200c")) != 0):
                    word_holder.insert(j, word)
                    j = j + 1
            list_of_tokens.clear()
            sent_holder_for_each_word.insert(i, word_holder)
            word_holder.clear()
            i = i+1
            # list_of_tokens = [x.strip("\u200c")
            #           for x in list_of_tokens if len(x.strip("\u200c")) != 0]
        return sent_holder_for_each_word

    def load_as_list_Farsi(self):
        with codecs.open('data/test.txt', encoding='UTF-8') as f:
            lines = f.readlines()
            # for line in f:
            #     vocab.append(line.strip().split())
        return lines


def main():
    myobj = Tokenizer()
    text = myobj.load_as_list_Farsi()
    sents = myobj.sent_tokenizer(str(text))
    words = myobj.word_tokenizer(sents)
    print(words[0])


if __name__ == '__main__':
    main()
