from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class CosineSimilarityEmbedder():
    
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        
    def build_index(self, file_names, separator = '$'):
        self.documents = []
        for file_name in file_names:
            with open(file_name, 'r') as file:
                self.documents.extend(file.read().split(separator))
        self.embeddings = self.model.encode_document(self.documents).astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
    def search(self, query, n_results):
        embedding = self.model.encode_query(query)
        embedding = embedding if len(embedding.shape) == 2 else np.array(embedding)
        faiss.normalize_L2(embedding)
        return  [[self.documents[i] for i in self.index.search(embedding, n_results)[1][j]] for j in range(len(query))]
        
    def save_index(self, file_name):
        with open(file_name, 'bw') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_index(file_name):        
        with open(file_name, 'br') as file:
            embedder = pickle.load(file)
        return embedder
        
# model = SentenceTransformer("./all-MiniLM-L6-v2")
# embedder = CosineSimilarityEmbedder(model)
# embedder.build_index(["texte_test.txt"])
# embedder.save_index("test_embedder")

embedder = CosineSimilarityEmbedder.load_index("test_embedder")
print(embedder.search(["Paris", "football", "Science"], 2))

