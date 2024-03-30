from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Define your sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaps over a slow-moving canine.",
    "The sky is clear and the stars are twinkling.",
    "Programming languages such as Python and Java are essential for software development.",
    "Economic indicators suggest a downturn in the market.",
    "Foxes are known for their quick movements and agility."
]

# Vectorize the sentences using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

# Calculate cosine similarity with the first sentence
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

# Reduce dimensions to 2D for visualization using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(tfidf_matrix.toarray())

# Plot the sentences in the reduced space
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(reduced):
    plt.scatter(x, y, color='blue' if i == 0 else 'green')
    plt.text(x+0.01, y+0.01, f"Sentence {i+1}", fontsize=9)

plt.title('2D Visualization of Sentence Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

# Print cosine similarity scores for review
for i, score in enumerate(cosine_sim[0], 1):
    print(f"Cosine similarity with Sentence {i}: {score}")
