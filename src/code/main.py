import ir_datasets
import numpy as np
import query
import lsi_model 
from sklearn.metrics.pairwise import cosine_similarity
import precomputer

dataset=ir_datasets.load('cranfield')

precomputed = precomputer.Precomputed(dataset=dataset)

lsi=lsi_model.Lsi_model(precomputed.corpus)

q = query.Query('data analysis',lsi)

similarities = cosine_similarity(q.vector, lsi.lsa.transform(lsi.X))

# Ordenar los documentos por similitud
most_similar_indices = np.argsort(similarities[0])[::-1]

# Mostrar los documentos más similares a la consulta
print("Documentos más similares a la consulta:")
for idx in most_similar_indices[:4]:
    print(f"- {precomputed.corpus[idx][:100]}...")
