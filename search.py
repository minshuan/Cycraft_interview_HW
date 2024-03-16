import tinydb
import argparse
import timeit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get sequence and limit")
    parser.add_argument("--query-sentence", type=str, help="Input string")
    parser.add_argument("--limit", type=int, help="Input integer")
    parser.add_argument("--time", action="store_true", help="Compare the speed of the two search methods")
    args = parser.parse_args()

    db = tinydb.TinyVectorDB()
    # db.show()

    input_str = args.query_sentence
    limit = args.limit
    str_vec = tinydb.BCE(input_str)

    top_sim_hnsw = db.search_similar_hnsw(str_vec, limit)

    print("Input sentence:",input_str)
    for list in top_sim_hnsw:
        for tpl in list:
            similarity, document = tpl
            print(f"Cosine similarity: {similarity}")
            print("Document:",document)
            print()
    if args.time:
        setup = f"from __main__ import db, str_vec, limit"
        time_taken_hnsw = timeit.timeit("db.search_similar_hnsw(str_vec, limit)", setup=setup, number=1000)
        time_taken = timeit.timeit("db.search_sim(str_vec, limit)", setup=setup, number=1000)
        print("Use hnsw search:%.8f"%time_taken_hnsw)
        print("Use loop search:%.8f"%time_taken)