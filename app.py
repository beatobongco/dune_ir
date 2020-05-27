from flask import Flask, render_template, request
from search import TfidfSearch, BM25Search, load_corpus

app = Flask(__name__)

corpus = load_corpus("dune.txt")
tfidf_search = TfidfSearch(corpus)
bm25_search = BM25Search(corpus)


@app.route("/")
@app.route("/bm25")
def index():
    query = request.args.get("query", "")
    results = bm25_search.search(query) if query else []
    return render_template("search.html", query=query, results=results, prefix="BM25")


@app.route("/tfidf")
def tfidf():
    query = request.args.get("query", "")
    results = tfidf_search.search(query) if query else []
    return render_template("search.html", query=query, results=results, prefix="TF-IDF")
