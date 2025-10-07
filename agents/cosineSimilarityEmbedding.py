from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import agents.base as base

class Embedder(base.Embedder):
    
    def __init__(self, model):
        self.model = model
        
    def get_embedding(self, text):
        if isinstance(text, list):
            return self.model.encode(text)
        return self.model.encode([text])[0]

    def tokenize(self, text):
        if isinstance(text, list):
            return self.model.tokenize(text)
        return self.model.tokenize([text])
    
    def embed(self):
        return None

class CosineSimilarityIndex(base.RAG):
    
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        
    def initialize(self, file_names, separator = '$'):
        self.documents = []
        for file_name in file_names:
            with open(file_name, 'r') as file:
                self.documents.extend(file.read().split(separator))
        self.embeddings = self.embedder.get_embedding(self.documents).astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
    def retrieve(self, query, n_results):
        embedding = self.embedder.get_embedding(query)
        embedding = embedding if len(embedding.shape) == 2 else np.array(embedding)
        faiss.normalize_L2(embedding)
        return  [[self.documents[i] for i in self.index.search(embedding, n_results)[1][j]] for j in range(len(query))]
        
    def save_index(self, file_name):
        with open(file_name, 'bw') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_index(file_name):        
        with open(file_name, 'br') as file:
            index = pickle.load(file)
        return index
        
model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = Embedder(model)
index = CosineSimilarityIndex(embedder)
index.initialize(["texte_test.txt"])
index.save_index("saved_models/test_embedder")

index = CosineSimilarityIndex.load_index("saved_models/test_embedder")

print(index.retrieve(["Paris", "football", "Science"], 2))

