
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, ne_chunk_sents
import nltk


class InformationExtraxtion:
    # def __init__(self, data):
    #     self.data = data

    def tokenizer(self, sample):
        chunk_store = []
        sentences = sent_tokenize(sample)
        tokenized_sentences = [word_tokenize(
            sentence) for sentence in sentences]
        tagged_sentences = [pos_tag(sentence)
                            for sentence in tokenized_sentences]

        for chunk in tagged_sentences:
            chunk_store.append(ne_chunk(chunk, binary=False))

        # for chunk in tagged_sentences:
        #     for i in range(1, len(chunk)):
        #         temp = ne_chunk(chunk[i], binary=False)
        #         chunk_store.append(temp)

        return chunk_store
