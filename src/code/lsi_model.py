from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Lsi_model:
    def __init__(self,corpus) -> None:
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(corpus)

        # Realizar descomposición en valores singulares (SVD)
        n_topics = 200  # Número de temas (conceptos) deseados
        self.lsa = TruncatedSVD(n_components=n_topics)
        self.lsa.fit(self.X)

