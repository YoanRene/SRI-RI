import gensim
import precomputer

precomputed = precomputer.Precomputed()

similarity_scores = [gensim.matutils.cossim(query_.tfidf, doc) for doc in precomputed.model.vector_repr]

ranking_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)

for idx in ranking_indices[:11]:
    print(f"Song = '{precomputed.df['song'][idx]}' Similarity = {similarity_scores[idx]}")    