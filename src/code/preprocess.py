import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(query, boolean = False):
    # Convertir a minúsculas
    query = query.lower()

    # Eliminar caracteres no alfanuméricos
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)

    # Tokenización
    tokens = word_tokenize(query)

    # Eliminar palabras vacías (stop words)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Unir tokens en una cadena de texto nuevamente
    preprocessed_query = (' & ' if boolean else ' ').join(tokens)

    return preprocessed_query