import os
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def read_corpus(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found at {file_path}. Please place it there before running.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text

def tokenize(text):
    return text.split()

def build_vocab(tokens, min_count=1):
    word_counts = collections.Counter(tokens)
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_count}
    return vocab, word_counts

def generate_training_data(tokens, vocab, window_size=2):
    training_pairs = []
    indices = [vocab[word] for word in tokens if word in vocab]
    for center_pos in range(len(indices)):
        center_word = indices[center_pos]
        start = max(0, center_pos - window_size)
        end = min(len(indices), center_pos + window_size + 1)
        for pos in range(start, end):
            if pos != center_pos:
                context_word = indices[pos]
                training_pairs.append((center_word, context_word))
    return training_pairs

class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = np.random.uniform(-0.8, 0.8, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-0.8, 0.8, (embedding_dim, vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, x):
        h = self.W1[x]
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        return y_pred, h, u

    def backward(self, x, y, y_pred, h, learning_rate):
        e = y_pred - y
        dW2 = np.outer(h, e)
        dW1 = np.dot(self.W2, e)
        self.W1[x] -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    def train(self, training_pairs, epochs=10, learning_rate=0.025):
        losses = []
        for epoch in range(epochs):
            loss = 0
            for center, context in training_pairs:
                y = np.zeros(self.vocab_size)
                y[context] = 1
                y_pred, h, u = self.forward(center)
                loss -= np.log(y_pred[context] + 1e-7)
                self.backward(center, y, y_pred, h, learning_rate)
            epoch_loss = loss / len(training_pairs)
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        return losses

    def get_embeddings(self):
        return self.W1

def plot_embeddings(embeddings, vocab, top_n=50, save_path=None):
    pca = PCA(n_components=2)
    embed_2d = pca.fit_transform(embeddings[:top_n])
    words = list(vocab.keys())[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(embed_2d[i, 0], embed_2d[i, 1]))
    plt.title("Word Embeddings visualized with PCA")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()