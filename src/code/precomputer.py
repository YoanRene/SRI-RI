import gensim
from cleaner import remove_stopwords,remove_noise,tokenice
from nltk.tokenize import word_tokenize
class Precomputed:
    def __init__(self,dataset):
        self.corpus=[(word_tokenize(doc.text),doc.doc_id) for doc in dataset.docs_iter()]
       # self.tokenized_docs = remove_stopwords(remove_noise(tokenice(self.corpus)))
       # self.dictionary = gensim.corpora.Dictionary(self.tokenized_docs)
       # self.corpus_in_bow = [self.dictionary.doc2bow(doc) for doc in self.tokenized_docs]
