import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sacrebleu

# Local imports
from src import word2vec as w2v
from src import transformer as tr

def run_word2vec():
    print("=== Task 1: Word Embeddings (Skip-Gram) ===")
    file_path = os.path.join("data", "word2vec_dataset.en")
    
    # Check if data exists
    if not os.path.exists(file_path):
        print(f"Skipping Q1: Place dataset at '{file_path}' to train.")
        return None, None

    text = w2v.read_corpus(file_path)
    tokens = w2v.tokenize(text)
    print(f"Total tokens: {len(tokens)}")

    vocab, word_counts = w2v.build_vocab(tokens, min_count=5)
    print(f"Initial vocab size: {len(vocab)}")

    filtered_tokens = [word for word in tokens if word in vocab]
    vocab, _ = w2v.build_vocab(filtered_tokens, min_count=1)
    print(f"Vocab size after rebuilding from filtered tokens: {len(vocab)}")

    training_pairs = w2v.generate_training_data(filtered_tokens, vocab, window_size=2)
    print(f"Training pairs generated: {len(training_pairs)}")

    embedding_dim = 50
    model = w2v.SkipGram(vocab_size=len(vocab), embedding_dim=embedding_dim)
    print("Training model...")
    model.train(training_pairs, epochs=10, learning_rate=0.025)

    embeddings = model.get_embeddings()
    os.makedirs("plots", exist_ok=True)
    w2v.plot_embeddings(embeddings, vocab, top_n=50, save_path=os.path.join("plots", "word2vec_pca.png"))
    print("Task 1 Complete. Plot saved to 'plots/word2vec_pca.png'.\n")

def run_transformer():
    print("=== Task 2: Transformer NMT ===")
    
    en_path = os.path.join("data", "Europarl Dataset", "europarl-v7.de-en.en")
    de_path = os.path.join("data", "Europarl Dataset", "europarl-v7.de-en.de")
    
    if not (os.path.exists(en_path) and os.path.exists(de_path)):
        print(f"Skipping Q2: Place English & German datasets under '{os.path.dirname(en_path)}' to train.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    en = tr.read_lines(en_path, limit=6000)
    de = tr.read_lines(de_path, limit=6000)
    
    data = list(zip(en, de))
    random.seed(42)
    random.shuffle(data)
    
    train_data = data[:5000]
    val_data = data[5000:6000]

    src_vocab = tr.build_vocab([s for s, _ in train_data])
    tgt_vocab = tr.build_vocab([t for _, t in train_data])
    inv_tgt_vocab = {i: t for t, i in tgt_vocab.items()}

    train_dataset = tr.TranslationDataset(train_data, src_vocab, tgt_vocab)
    val_dataset = tr.TranslationDataset(val_data, src_vocab, tgt_vocab)

    pad_idx = tgt_vocab["<pad>"]
    collate_fn = tr.PadCollate(pad_idx=pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    model = tr.SimpleTransformer(len(src_vocab), len(tgt_vocab), dim=128, heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print("Training Transformer Model...")
    train_losses = []
    epochs = 5
    for epoch in range(epochs):
        loss = tr.train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(loss)
        print(f"Epoch {epoch+1} loss: {loss:.4f}")

    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", "transformer_loss_curve.png"))
    plt.show()

    # Calculate BLEU Score
    references = []
    hypotheses = []
    for src_tensor, tgt_tensor in val_dataset:
        pred = tr.greedy_decode(model, src_tensor, inv_tgt_vocab, device)
        target = [inv_tgt_vocab.get(t.item(), "<unk>") for t in tgt_tensor[1:-1]]
        hypotheses.append(" ".join(pred))
        references.append([" ".join(target)])

    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"\nValidation BLEU score: {bleu.score:.2f}")
    print("Task 2 Complete. Loss Curve saved to 'plots/transformer_loss_curve.png'.\n")

if __name__ == "__main__":
    run_word2vec()
    run_transformer()