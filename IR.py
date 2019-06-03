from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk, ne_chunk_sents, re
import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class InformationExtraxtion:
    # def __init__(self, data):
    #     self.data = data

    def fuzzzy(self, source, query):
        # choices = process.extract(query, source)
        _top = process.extractOne(query, source)
        _bests = process.extractBests(query, source)
        return _bests

    def tokenizer(self, sample):
        chunk_store = []
        sentences = sent_tokenize(sample)
        tokenized_sentences = [word_tokenize(
            sentence) for sentence in sentences]
        tagged_sentences = [pos_tag(sentence)
                            for sentence in tokenized_sentences]

        for chunk in tagged_sentences:
            chunk_store.append(ne_chunk(chunk, binary=False))

        return chunk_store

    def relation_extraction(self, data):
        out = []

        class doc():
            pass

        IN = re.compile(r'.*\b in \b(?!\b.+ing)')
        doc.headline = ["test headline for sentence"]
        for sent in enumerate(data):
            doc.text = sent
            for rel in nltk.sem.extract_rels('PER', 'LOC', doc, corpus='ieer', pattern=IN):
                out.append(nltk.sem.rtuple(rel))
        return out
