class Metrics:
    def __init__(self,relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado,irrelevante_no_recuperado):
        self.precision_=self.precision(relevante_recuperado,irrelevante_recuperado)
        self.recall_=self.recall(relevante_recuperado,relevante_no_recuperado)
        self.f1_=self.f1(relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado)
        self.fallout_=self.fallout(irrelevante_recuperado,irrelevante_no_recuperado)

    def precision(self,relevante_recuperado,irrelevante_recuperado):
        try:
            return len(relevante_recuperado)/len(set(relevante_recuperado).union(set(irrelevante_recuperado)))
        except:
            return -1
    
    def recall(self,relevante_recuperado,relevante_no_recuperado):
        try:
            return len(relevante_recuperado)/len(set(relevante_recuperado).union(set(relevante_no_recuperado)))
        except:
            return -1

    def f1(self,relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado):
        try:
            p=self.precision(relevante_recuperado,irrelevante_recuperado)
            r=self.recall(relevante_recuperado,relevante_no_recuperado)
            return (2*p*r)/(p+r)
        except:
            return -1

    def fallout(self,irrelevante_recuperado,irrelevante_no_recuperado):
        try:
            return len(irrelevante_recuperado)/len(set(irrelevante_recuperado).union(set(irrelevante_no_recuperado)))
        except:
            return -1

    