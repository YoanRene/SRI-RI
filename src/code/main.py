import ir_datasets
import numpy as np
import query
import metrics
import lsi_model 
from sklearn.metrics.pairwise import cosine_similarity
import precomputer

dataset=ir_datasets.load('cranfield')

precomputed = precomputer.Precomputed(dataset=dataset)

lsi=lsi_model.Lsi_model(precomputed.corpus)
doc_ids=[doc.doc_id for doc in dataset.docs_iter()]
dic={}
for qrel in dataset.qrels_iter():
    if qrel.relevance>=3:
        if qrel.query_id not in dic:
            dic[qrel.query_id]=list()
        dic[qrel.query_id].append(qrel.doc_id)



dic_queries={}
for query_ in dataset.queries_iter():
    #print(query_.query_id)
    if query_.query_id not in dic: 
        continue

    q = query.Query(query_.text,lsi)

    similarities = cosine_similarity(q.vector, lsi.lsa.transform(lsi.X))

    # Ordenar los documentos por similitud
    most_similar_indices = np.argsort(similarities[0])[::-1]
    most_similar_indices_filtered = [j for j in most_similar_indices if similarities[0][j]>=0.6]
    dic_queries[query_.query_id]=[str(i) for i in most_similar_indices_filtered]
    # Mostrar los documentos más similares a la consulta
    #print("Documentos más similares a la consulta:")
    #for idx in most_similar_indices_filtered[:4]:
    #    print(f"- {precomputed.corpus[idx][:10]} similitud: {similarities[0][idx]}")


for i in dic_queries.keys():
    recuperado=dic_queries[i]
    relevante=dic[i]
    relevante_recuperado=set(recuperado).intersection(set(relevante))
    relevante_no_recuperado=set(relevante).difference(set(recuperado))
    irrelevante_recuperado=set(recuperado).difference(set(relevante))
    irrelevante_no_recuperado=(set(doc_ids).difference(set(recuperado))).difference(set(relevante))
    print('   ')
    print(len(relevante_recuperado))
    print(len(relevante_no_recuperado))
    print(len(irrelevante_recuperado))
    print(len(irrelevante_no_recuperado))
    metric=metrics.Metrics(relevante_recuperado,irrelevante_recuperado,relevante_no_recuperado,irrelevante_no_recuperado)
    print(f'P:{metric.precision_}')
    print(f'R:{metric.recall_}')
    print(f'F1:{metric.f1_}')
    print(f'Fal:{metric.fallout_}')
