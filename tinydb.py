from BCEmbedding import EmbeddingModel
import pandas as pd
import numpy as np
import os
import nmslib
import pickle

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
            self.load_lsh_index()
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
        self.hnsw_index.addDataPointBatch(self.data['EmbVector'].tolist())
        self.hnsw_index.createIndex({'post': 2}, print_progress=True)
        self.lsh_index = LSHIndex(self.data)
        self.lsh_index.build_index()
        self.save_hnsw_index()
    def save_hnsw_index(self):
        self.hnsw_index.saveIndex('hnsw_index.bin')
        print("hnsw_index save")        
    def load_hnsw_index(self):
        if os.path.exists('hnsw_index.bin'):
            self.hnsw_index = nmslib.init(method='hnsw', space='cosinesimil')
            self.hnsw_index.loadIndex('hnsw_index.bin')
            print("load hnsw_index")
        else:
            self.build_hnsw_index()
# add lsh
    def build_lsh_index(self):
        self.lsh_index = LSHIndex(self.data)
        self.lsh_index.build_index()
        self.save_lsh_index()
    def save_lsh_index(self):
        with open('lsh_index.bin', 'wb') as f:
            pickle.dump(self.lsh_index, f)
        print("lsh_index save") 
    def load_lsh_index(self):
        if os.path.exists('lsh_index.bin'):
            with open('lsh_index.bin', 'rb') as f:
                self.lsh_index = pickle.load(f)
            print("load lsh_index")            
        else:
            self.build_lsh_index()

    def search_similar_hnsw(self, query_vector, n):
        idxs, _ = self.hnsw_index.knnQuery(query_vector, 2*n)
        similarities = []
        for i in idxs:
            sim = cosine_sim(query_vector, self.data['EmbVector'].iloc[i])
            similarities.append((float(sim), self.data.iloc[i]['Document']))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = [[(sim, doc) for sim, doc in similarities[:n]]]
        return top_n

    def search_similar(self, str_vec, n):
        similarities = []
        for i, vector in enumerate(self.data['EmbVector']):
            sim = cosine_sim(str_vec, vector)
            similarities.append((sim, self.data.iloc[i]['Document']))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = [[(sim, doc) for sim, doc in similarities[:n]]]
        return top_n    
    
    def search_similar_lsh(self, query_vector, n):
        candidate = self.lsh_index.search(query_vector, n)
        similarities = []
        for idx in candidate:
            sim = cosine_sim(query_vector, self.data['EmbVector'].iloc[idx])
            similarities.append((sim, self.data.iloc[idx]['Document']))
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_n = [[(sim, doc) for sim, doc in similarities[:n]]]
        return top_n

    def save(self):
        self.data.to_json('data.json', orient='records')
        print("data save")

    def show(self):
        print(self.data)
    
class LSHIndex:
    def __init__(self, data):
        self.data = data
        self.num_tables = 1
        self.num_hashes = 1
        self.num_dimensions = 768
        self.lsh_index = [{} for _ in range(self.num_tables)]
        self.hyperplanes = [np.random.randn(self.num_dimensions, self.num_hashes) for _ in range(self.num_tables)]

    def hash_vector(self, vector):
        hash_buckets = []
        for i in range(self.num_tables):
            hash_values = ''
            for j in range(self.num_hashes):
                hash_val = np.dot(vector, self.hyperplanes[i][:, j]) > 0
                hash_values += str(int(hash_val))
            hash_buckets.append(hash_values)
        return hash_buckets
    
    def build_index(self):
        for index, vector in enumerate(self.data['EmbVector']):
            hashes = self.hash_vector(vector)
            for table, hash_value in enumerate(hashes):
                if hash_value not in self.lsh_index[table]:
                    self.lsh_index[table][hash_value] = []
                self.lsh_index[table][hash_value].append(index)

    def search(self, query_vector, n):
        query_buckets = self.hash_vector(query_vector)
        candidate_idxs = set()
        for table, hash_value in enumerate(query_buckets):
            if hash_value in self.lsh_index[table]:
                candidate_idxs.update(self.lsh_index[table][hash_value])
        return candidate_idxs

# db = TinyVectorDB()
# db.load_index()
# db.show()