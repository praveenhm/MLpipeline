import numpy as np

class BinaryVectorSearch:
    def __init__(self, data):
        self.data = data
        self.num_vectors, self.num_dimensions = data.shape
        self.threshold = np.mean(data)
        self.binary_data = self.float_to_binary(data)

    def float_to_binary(self, vector):
        return np.where(vector > self.threshold, 1, 0)

    def binary_to_float(self, vector):
        return vector.astype(float)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def binary_cosine_similarity(self, vec1, vec2):
        intersection = np.sum(vec1 & vec2)
        union = np.sum(vec1 | vec2)
        return intersection / union

    def adaptive_retrieval(self, query_vector, top_k, rerank_factor=2):
        binary_query = self.float_to_binary(query_vector)
        binary_similarities = np.apply_along_axis(self.binary_cosine_similarity, 1, self.binary_data, binary_query)
        candidate_indices = np.argpartition(binary_similarities, -rerank_factor * top_k)[-rerank_factor * top_k:]

        candidate_vectors = self.data[candidate_indices]
        float_query = self.binary_to_float(binary_query)
        float_similarities = np.apply_along_axis(self.cosine_similarity, 1, candidate_vectors, float_query)
        top_indices = candidate_indices[np.argsort(-float_similarities)][:top_k]

        return top_indices

    def search(self, query_vector, top_k=100, rerank_factor=2):
        top_indices = self.adaptive_retrieval(query_vector, top_k, rerank_factor)
        top_vectors = self.data[top_indices]
        top_similarities = np.apply_along_axis(self.cosine_similarity, 1, top_vectors, query_vector)

        return top_indices, top_similarities

def main():
    # Example data
    num_vectors = 10000
    num_dimensions = 100
    data = np.random.randn(num_vectors, num_dimensions)
    print(f"Generated {num_vectors} vectors with {num_dimensions} dimensions each.")

    # Create BinaryVectorSearch instance
    binary_search = BinaryVectorSearch(data)
    print(f"Converted floating-point vectors to binary vectors using threshold: {binary_search.threshold:.4f}")

    # Example query vector
    query_vector = np.random.randn(num_dimensions)
    print("Generated a random query vector.")

    # Perform search
    top_k = 100
    rerank_factor = 2
    top_indices, top_similarities = binary_search.search(query_vector, top_k, rerank_factor)

    # Print the indices and similarities of the top 5 most similar vectors
    top_n = 5
    print(f"\nTop {top_n} most similar vectors:")
    for i, (index, similarity) in enumerate(zip(top_indices[:top_n], top_similarities[:top_n]), 1):
        print(f"Vector {i}:")
        print(f"  Index: {index}")
        print(f"  Floating-point Similarity: {similarity:.4f}")
        print(f"  Vector: {binary_search.data[index]}")

if __name__ == "__main__":
    main()