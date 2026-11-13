import numpy as np


class VectorStore:

    def __init__(self):
        self.items = []

    def add(self, chunk, vector):
        self.items.append(
            {
                "chunk": chunk,
                "vector": vector
            }
        )

    def count(self):
        return len(self.items)

    def search(self, query_vector, top_k=3):

        results = []

        for item in self.items:

            similarity = self.cosine_similarity(
                query_vector,
                item["vector"]
            )

            results.append(
                {
                    "chunk": item["chunk"],
                    "score": similarity
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]

    @staticmethod
    def cosine_similarity(vector_a, vector_b):

        a = np.array(vector_a)
        b = np.array(vector_b)

        return np.dot(a, b) / (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )