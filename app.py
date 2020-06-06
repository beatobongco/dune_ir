from flask import Flask, render_template, request, jsonify
from search import TfidfSearch, BM25Search, load_corpus

app = Flask(__name__)

corpus = load_corpus("dune.txt")
tfidf_search = TfidfSearch(corpus)
bm25_search = BM25Search(corpus)


@app.route("/")
@app.route("/tfidf")
def index():
    query = request.args.get("query", "")
    results = tfidf_search.search(query) if query else []
    return render_template("search.html", query=query, results=results, prefix="TF-IDF")


@app.route("/bm25")
def bm25():
    query = request.args.get("query", "")
    results = bm25_search.search(query) if query else []
    return render_template("search.html", query=query, results=results, prefix="BM25")


@app.route("/api/bm25", methods=["POST"])
def bm25_api():
    print(request.get_json())
    query = request.get_json().get("query")
    results = bm25_search.search(query) if query else []
    return jsonify(results)
