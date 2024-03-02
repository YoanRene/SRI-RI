try:
    import nltk
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nltk"])
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

def tokenice(corpus):
    return [nltk.tokenize.word_tokenize(doc) for doc in corpus]

def remove_noise(tokenized_docs):
    return [[word.lower() for word in doc if word.isalpha()] for doc in tokenized_docs]

def remove_stopwords(tokenized_docs):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [[word for word in doc if word not in stop_words] for doc in tokenized_docs]