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
        similarity_scores = cosine_similarity(
            query_vector, self.document_vectors
        ).flatten()

        # sort indices then flip so that highest scores first
        sorted_indices = np.flip(np.argsort(similarity_scores))

        # get top k scores
        top_k = self.corpus[sorted_indices].flatten()[:k]
        top_k_scores = similarity_scores[sorted_indices][:k]

        return list(zip(top_k, top_k_scores))


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
