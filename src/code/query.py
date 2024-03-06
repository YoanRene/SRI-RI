import cleaner

class Query:
    def __init__(self,query,model):
        self.query=query
        #query_bow = model.dictionary.doc2bow(self.query.lower().split())
        self.vector = model.lsa.transform(model.vectorizer.transform([query]))