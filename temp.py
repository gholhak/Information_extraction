import nltk

sents = nltk.corpus.treebank.tagged_sents()
_bank = []
for se in sents:
    nltk.ne_chunk(se, binary=True)
    _bank.append(_bank)