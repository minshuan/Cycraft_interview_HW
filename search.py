import tinydb
import argparse
import timeit
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get sequence and limit")
    parser.add_argument("--query-sentence", type=str, help="Input string")
    parser.add_argument("--limit", type=int, help="Input integer", default=1)
    parser.add_argument("--time", action="store_true", help="Compare the speed of the two search methods")
    args = parser.parse_args()

    
    db = tinydb.TinyVectorDB()
    # db.show()
    if not db.data.empty:
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
            time_taken_hnsw = timeit.timeit("db.search_similar_hnsw(str_vec, limit)", setup=setup, number=100)
            time_taken = timeit.timeit("db.search_similar(str_vec, limit)", setup=setup, number=100)
            time_taken_lsh = timeit.timeit("db.search_similar_lsh(str_vec, limit)", setup=setup, number=100)

            print("Use hnsw search:%.8f"%time_taken_hnsw)
            print("Use loop search:%.8f"%time_taken)
            print("Use lsh search:%.8f"%time_taken_lsh)
    else:
        print("No data can be searched.")