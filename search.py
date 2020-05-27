import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bm25 import BM25

def load_corpus(path):
    """Load the text of the Dune book, naively split into paragraphs
    
    Returns np.array of the paragraphs"""

    with open("dune.txt") as f:
        dune_text = f.read()

    dune_paragraphs = []
    for p in dune_text.split("\n\n"):
        _p = p.strip()
        # Ignore useless lines
        if len(_p) > 1:
            dune_paragraphs.append(_p)

    return np.array(dune_paragraphs)

class TfidfSearch:
    def __init__(self, corpus):
        # our paragraphs
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        # paragraph vectors
        self.document_vectors = self.vectorizer.fit_transform(corpus)
    
    def search(self, query, k=10):
        # 1 x vocab_size
        query_vector = self.vectorizer.transform([query])
        
        # cos_sim(1 x vocab_size, n_paragraphs x vocab_size) = 1 x n_paragraphs
        similarity_scores = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # sort indices then flip so that highest scores first
        sorted_indices = np.flip(np.argsort(similarity_scores))

        # get top k scores
        top_k = self.corpus[sorted_indices].flatten()[:k]
        top_k_scores = similarity_scores[sorted_indices][:k]
        
        return list(zip(top_k, top_k_scores))

# dune_paragraphs = load_corpus("dune.txt")

# tfidf_search = TfidfSearch(dune_paragraphs)
# vectorizer = TfidfVectorizer()
# paragraph_vectors = vectorizer.fit_transform(dune_paragraphs)

# # vectorizer is now trained!
# def tfidf_search(query, k=10):
#     # 1 x vocab_size
#     query_vector = vectorizer.transform([query])
#     # cos_sim(1 x vocab_size, n_paragraphs x vocab_size) = 1 x n_paragraphs
#     similarity_scores = cosine_similarity(query_vector, paragraph_vectors).flatten()
#     sorted_indices = np.flip(np.argsort(similarity_scores))
#     top_k = dune_paragraphs[sorted_indices].flatten()[:k]
#     top_k_scores = similarity_scores[sorted_indices][:k]
    
#     res = []
#     for paragraph, score in zip(top_k, top_k_scores):
#         res.append((idx + 1, paragraph, score))
#     return res 

# q = "Dune desert planet"
# res = tfidf_search.search(q)
# desert power
# leto atreides
# worm

# TODO: demonstrate BM25

class BM25Search:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = BM25()
        self.vectorizer.fit(corpus)

    def search(self, query, k=10):
        similarity_scores = self.vectorizer.transform(query, self.corpus)
        sorted_indices = np.flip(np.argsort(similarity_scores))
        top_k = self.corpus[sorted_indices].flatten()[:k]
        top_k_scores = similarity_scores[sorted_indices][:k]
        return list(zip(top_k, top_k_scores))

# bm25_vectorizer = BM25()
# bm25_vectorizer.fit(dune_paragraphs)
# # bm25_par_vectors = bm25_vectorizer.transform(dune_paragraphs)

# def bm25_search(query, k=10):
#     # 1 x vocab_size
#     similarity_scores = bm25_vectorizer.transform(query, dune_paragraphs)
#     # cos_sim(1 x vocab_size, n_paragraphs x vocab_size) = 1 x n_paragraphs
#     # similarity_scores = cosine_similarity(query_vector, bm25_par_vectors).flatten()
#     sorted_indices = np.flip(np.argsort(similarity_scores))
#     top_k = dune_paragraphs[sorted_indices].flatten()[:k]
#     top_k_scores = similarity_scores[sorted_indices][:k]
#     res = []
#     for idx, (paragraph, score) in enumerate(zip(top_k, top_k_scores)):
#         res.append((idx + 1, paragraph, score))
#     return res 

# res2 = bm25_search(q)