import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=1024):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add(self, chunks, vectors):
        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.chunks.extend(chunks)

    def search(self, query_vector, k=4):
        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            results.append({**self.chunks[i], "score": float(score)})
        return results