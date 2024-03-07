from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from preprocess import preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class Lsi_model:
    def __init__(self, corpus) -> None:
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=500)
        corpus = [corpus for corpus,_ in corpus]
        corpus = [' '.join(doc) for doc in corpus]
        self.X = self.vectorizer.fit_transform(corpus)

        # Realizar descomposición en valores singulares (SVD)
        n_topics = 2  # Ajustar el número de temas según sea necesario
        self.lsa = TruncatedSVD(n_components=n_topics)
        self.U = self.lsa.fit_transform(self.X)
    
    def _get_similarities(self,query):
        preprocessed_query = preprocess(query)

        # Vectorización de la consulta
        query_vector = self.vectorizer.transform([preprocessed_query])

        # Transformación de la consulta en el espacio de temas
        query_topic = self.lsa.transform(query_vector)


        return cosine_similarity(query_topic, self.U)
    def  get_most_similar(self,query,count = 10):
        similarities = self._get_similarities(query)

        # Ordenar los documentos por similitud
        most_similar_indices = np.argsort(similarities[0])[::-1]
        return most_similar_indices[:10]
    def get_filtered_docs(self,query,umbral=0.6):
        similarities = self._get_similarities(query)

        # Ordenar los documentos por similitud
        most_similar_indices = np.argsort(similarities[0])[::-1]
        most_similar_indices_filtered = [j for j in most_similar_indices if similarities[0][j]>=umbral]
        return most_similar_indices_filtered

