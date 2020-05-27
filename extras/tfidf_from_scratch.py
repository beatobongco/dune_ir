import math
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import normalize


class ScratchTFIDFVectorizer:
    """TFIDF vectorizer from scratch
    
        vect = ScratchTFIDFVectorizer()
        X = vect.fit_transform(corpus)
    """

    def __init__(self):
        # check if not
        nltk.download("punkt")
        self.corpus_counter = None
        self.vocab = []
        self.total_docs = 0

    def fit_transform(self, corpus):
        """Fits the vectorizer on a corpus, converting text into TF-IDF vector representations
        """
        # get vocab and tokenized forms
        tokenized_documents = []
        # counts number of documents a word appears in
        corpus_counter = Counter()
        for doc in corpus:
            wt = word_tokenize(doc)
            tokenized_documents.append(wt)
            # get unique words in each document, count how many times they appear
            # this will be used later on for inverse document frequency
            corpus_counter.update(set(wt))

        self.corpus_counter = corpus_counter
        self.vocab = list(corpus_counter.keys())
        self.total_docs = len(tokenized_documents)

        return self.transform(corpus)

    def transform(self, documents):
        """Transforms documents in TFIDF vectors 
        """
        vectors = []
        for document in documents:
            document = word_tokenize(document)
            vector = np.zeros(len(self.vocab))
            doc_counter = Counter()
            doc_counter.update(document)

            # term-document matrix
            for doc_word, doc_count in doc_counter.items():
                if doc_word not in self.vocab:
                    continue

                # term frequency
                # (Number of times term t appears in a document) / (Total number of terms in the document)
                tf = doc_count / len(document)
                vector[self.vocab.index(doc_word)] += tf * self.idf(doc_word)

            vectors.append(vector)
        return normalize(vectors, norm="l2")

    def idf(self, word):
        # inverse document frequency
        # log(Total number of documents / Number of documents with term t in it).
        # we add 1 to the denominator for smoothing
        return (
            math.log((1 + self.total_docs) / (1 + self.corpus_counter.get(word, 0))) + 1
        )
