import tinydb
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get sequence and limit")
    parser.add_argument("--query-sentence", type=str, help="Input string")
    parser.add_argument("--limit", type=int, help="Input integer")
    args = parser.parse_args()

    db = tinydb.TinyVectorDB()
    # db.show()

    input_str = args.query_sentence
    limit = args.limit
    str_vec = tinydb.BCE(input_str)
    top_sim_hnsw = db.search_similar_hnsw(str_vec, limit)
    top_sim = db.search_sim(str_vec, limit)
    print("Input sentence:",input_str)
    print("Top %d similar documents and there Cosine similarity:"%limit)
    for Document, similarity in top_sim_hnsw:
        print(f"Document: {Document}, Cosine similarity: {similarity}")
    print("Top %d similar documents and there Cosine similarity:"%limit)
    for Document, similarity in top_sim:
        print(f"Document: {Document}, Cosine similarity: {similarity}")