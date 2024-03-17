
from BCEmbedding import EmbeddingModel
import pandas as pd
import numpy as np
import os
import nmslib

def BCE(sequence):
    model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")
    embeddings = model.encode(sequence)
    return embeddings

def cosine_sim(vector_1, vector_2):
    sim = vector_1.dot(vector_2.T) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return sim

class TinyVectorDB:

    def __init__(self):
        if os.path.exists('data.json'):
            self.data = pd.read_json('data.json')
            print("load data")
            self.data['EmbVector'] = self.data['EmbVector'].apply(np.array)
            self.load_hnsw_index()
        else:
            self.data = pd.DataFrame(columns=['Document', 'EmbVector'])
            self.hnsw_index = None
            self.lsh_index = None

    # use model embedding->768
    def insert(self, sequence):
        vector = BCE(sequence)
        # print(vector.shape)
        self.data = self.data._append({'Document': sequence, 'EmbVector': vector}, ignore_index=True)
    
    def build_hnsw_index(self):
        self.hnsw_index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(self.data['EmbVector'].tolist())
        self.index.createIndex({'post': 2}, print_progress=True)
        self.save_hnsw_index()
    def save_hnsw_index(self):
        self.index.saveIndex('hnsw_index.bin')
        print("hnsw_index save")
    def load_hnsw_index(self):
        if os.path.exists('hnsw_index.bin'):
            self.hnsw_index = nmslib.init(method='hnsw', space='cosinesimil')
            self.hnsw_index.loadIndex('index.bin')
            print("load hnsw_index")
        else:
            self.build_hnsw_index()

    def search_similar_hnsw(self, query_vector, n):
        idxs, _ = self.hnsw_index.knnQuery(query_vector, 2*n)
        similarities = []
        for i in idxs:
            sim = cosine_sim(query_vector, self.data['EmbVector'].iloc[i])
            similarities.append((float(sim), self.data.iloc[i]['Document']))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = [[(sim, doc) for sim, doc in similarities[:n]]]
        return top_n

    def search_sim(self, str_vec, n):
        similarities = []
        for i, vector in enumerate(self.data['EmbVector']):
            sim = cosine_sim(str_vec, vector)
            similarities.append((sim, self.data.iloc[i]['Document']))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = [[(sim, doc) for sim, doc in similarities[:n]]]
        return top_n    
    
    def save(self):
        self.data.to_json('data.json', orient='records')
        print("data save")

    def show(self):
        print(self.data)
    

# db = TinyVectorDB()
# db.load_index()
# db.show()