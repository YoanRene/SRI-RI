import reader
import os
from cleaner import remove_stopwords,remove_noise,tokenice

class Precomputed:
    def __init__(self,size=50,dataset_path=os.getcwd()+'\\data\\spotify_millsongdata.csv',duplicated_subset='song'):
        self.df = reader.read(size,dataset_path,duplicated_subset)
        self.df['merged'] = self.df['artist']+" "+self.df['song']+" "+self.df['text']
        corpus= [doc for doc in self.df["merged"]]
        tokenized_docs = remove_stopwords(remove_noise(tokenice(corpus)))