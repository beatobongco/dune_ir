from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from search import TfidfSearch, BM25Search, load_corpus
from loguru import logger

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

corpus = load_corpus("dune.txt")
tfidf_search = TfidfSearch(corpus)
bm25_search = BM25Search(corpus)


def format_results(results):
    return {"hits": [{"text": res, "score": score} for res, score in results]}


def do_bm25_search(query, k1=1.2, b=0.75):
    """"""
    if query:
        logger.debug(query, k1, b)
        searcher = BM25Search(corpus, k1, b)
        results = searcher.search(query)
        return format_results(results)
    else:
        return []


@app.route("/")
@app.route("/tfidf")
def index():
    return render_template("search.html", prefix="TF-IDF")


@app.route("/bm25")
def bm25():
    return render_template("search.html", prefix="BM25")


@app.route("/api/search", methods=["POST"])
def bm25_api():
    rj = request.get_json()
    logger.debug(rj)
    query = rj.get("query")
    k1 = rj.get("k1")
    b = rj.get("b")
    search_type = rj.get("search_type")
    if search_type == "tfidf":
        logger.debug(
            f"TFIDF mode: {search_type}, feature_names: {len(tfidf_search.vectorizer.get_feature_names())}"
        )
        results = format_results(tfidf_search.search(query))
    else:
        results = do_bm25_search(query, k1=k1, b=b)
        logger.debug(results)
    return jsonify(results)
