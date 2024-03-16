
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
            self.load_index()
        else:
            self.data = pd.DataFrame(columns=['Document', 'EmbVector'])

    # use model embedding->768
    def insert(self, sequence):
        vector = BCE(sequence)
        # print(vector.shape)
        self.data = self.data._append({'Document': sequence, 'EmbVector': vector}, ignore_index=True)
    
    def build_index(self):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(self.data['EmbVector'].tolist())
        self.index.createIndex({'post': 2}, print_progress=True)
        self.save_index()
    def save_index(self):
        self.index.saveIndex('index.bin')
        print("index save")
    def load_index(self):
        if os.path.exists('index.bin'):
            self.index = nmslib.init(method='hnsw', space='cosinesimil')
            self.index.loadIndex('index.bin')
            print("load index")
        else:
            self.build_index()

    def search_similar_hnsw(self, query_vector, n):
        idxs, _ = self.index.knnQuery(query_vector, 2*n)
        similarities = []
        for i in idxs:
            sim = cosine_sim(query_vector, self.data['EmbVector'].iloc[i])
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_n = [(self.data.iloc[i]['Document'], sim) for i, sim in similarities[:n]]
        return top_n

    def search_sim(self, str_vec, n):
        similarities = []
        for i, vector in enumerate(self.data['EmbVector']):
            sim = cosine_sim(str_vec, vector)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_n = [(self.data.iloc[i]['Document'], sim) for i, sim in similarities[:n]]
        return top_n
    
    def save(self):
        self.data.to_json('data.json', orient='records')
        print("data save")

    def show(self):
        print(self.data)
    

# db = TinyVectorDB()
# db.load_index()
# db.show()