from word2vec import Word2vec
import numpy as np
import pandas as pd
from FarsiTokenizer import Tokenizer

settings = {}
settings['n'] = 5  # dimension of word embeddings
settings['window_size'] = 2  # context window +/- center word
settings['min_count'] = 0  # minimum word count
settings['epochs'] = 5000  # number of training epochs
settings['neg_samp'] = 10  # number of negative words to use during training
settings['learning_rate'] = 0.01  # learning rate
np.random.seed(0)  # set the seed for reproducibility

tk_obj = Tokenizer()
w2v = Word2vec(settings)

doc = 'data\\ner.txt.csv'

corpus = pd.read_csv(doc)

doc_holder = tk_obj.ner_data_document_extraction(corpus)
training_data = w2v.generate_training_data(doc_holder)
w2v.train(training_data)
print('log')