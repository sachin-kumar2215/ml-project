import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from dataclasses import dataclass, field
import os
import json
import pickle  # For saving Vocabulary
from typing import List, Dict, Any, Tuple

# Import your CNN model definition
from src.models.sentiment_model import CNNSentimentClassifier

# Import data processing functions
from src.data.processor import (
    DataPreprocessor,
    download_and_load_data,
    combine_comments,
    extract_and_process_comments,
    create_ensemble_sentiment,
)

# Set up device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Configuration Dataclasses ---
@dataclass
class TrainConfig:
    data_url: str = (
        "https://storage.googleapis.com/sachin_data1/_Social%20Media%20Analytics%20-%20LLM%20-%20Socila%20Media%20Analytics.csv"
    )
    local_raw_data_path: str = "data/raw/social_media_data.csv"
    processed_data_path: str = "data/processed/processed_social_media_data.csv"
    test_size: float = 0.2  # For train/val/test split (initial split)
    val_size: float = 0.5  # For val/test split from temp data
    random_state: int = 42
    num_epochs: int = 30
    batch_size: int = 64
    patience: int = 7  # Early stopping patience
    model_save_path: str = "models/best_cnn_model.pth"
    vocab_save_path: str = "models/vocabulary.pkl"
    label_map_save_path: str = "models/label_map.json"
    results_save_path: str = "models/training_results.json"


@dataclass
class SentimentConfig:
    # Vocabulary parameters
    min_freq: int = 2
    max_vocab_size: int = 10000
    max_seq_length: int = 100  # Will be dynamically set based on data (95th percentile)

    # CNN Model Hyperparameters
    embed_dim: int = 128
    num_filters: int = 100
    filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    dropout: float = 0.3
    lr: float = 0.001
    num_classes: int = 3  # Negative, Neutral, Positive


# --- Vocabulary Creation ---
class Vocabulary:
    """
    Manages the mapping between words and their integer indices.
    """

    def __init__(self, texts: List[str], min_freq: int = 2, max_vocab_size: int = None):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.build_vocab(texts)

    def build_vocab(self, texts: List[str]):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        idx = 2
        for word, freq in sorted_words:
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                if self.max_vocab_size and idx >= self.max_vocab_size:
                    break

    def encode(self, text: str) -> List[int]:
        """Converts a text string into a list of integer indices."""
        return [self.word2idx.get(word, 1) for word in text.split()]

    def decode(self, indices: List[int]) -> str:
        """Converts a list of integer indices back to a text string."""
        return " ".join(
            [self.idx2word.get(idx, "<UNK>") for idx in indices if idx != 0]
        )

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.word2idx)


# --- PyTorch Dataset & DataLoader ---
class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for sentiment classification.
    """

    def __init__(
        self, texts: List[str], labels: List[str], vocab: Vocabulary, max_length: int
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }  # Fixed mapping for consistency
        self.reverse_label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.vocab.encode(text)

        # Truncate or pad to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[: self.max_length]
        else:
            encoded.extend([0] * (self.max_length - len(encoded)))  # 0 is PAD

        return {
            "text": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(self.label_map[label], dtype=torch.long),
            "length": torch.tensor(
                min(len(encoded), self.max_length), dtype=torch.long
            ),  # Actual length after truncation
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader to handle batching.
    """
    texts = torch.stack([item["text"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])

    return {"texts": texts, "labels": labels, "lengths": lengths}


# --- Model Training Utilities ---
class ModelTrainer:
    """
    Handles the training and validation loops for the PyTorch model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch in self.train_loader:
            texts = batch["texts"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            outputs = self.model(texts, lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )  # Gradient clipping
            optimizer.step()

            total_loss += loss.item() * texts.size(0)
            total_samples += texts.size(0)

        return total_loss / total_samples

    def validate_epoch(self, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in self.val_loader:
                texts = batch["texts"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"]

                outputs = self.model(texts, lengths)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()

                total_loss += loss.item() * texts.size(0)
                total_samples += texts.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100

        return avg_loss, accuracy

    def train(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        patience: int = 5,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Train the model with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        print("\nStarting training...")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(optimizer, criterion)
            val_loss, val_accuracy = self.validate_epoch(criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(), "models/best_cnn_model.pth"
                )  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} due to no improvement in validation loss."
                    )
                    break

        self.model.load_state_dict(
            torch.load("models/best_cnn_model.pth")
        )  # Load best model
        print("Training finished. Best model loaded.")
        return self.train_losses, self.val_losses, self.val_accuracies


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    label_map: Dict[str, int],
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]

            outputs = model(texts, lengths)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1_weighted = f1_score(all_labels, all_predictions, average="weighted")
    f1_macro = f1_score(all_labels, all_predictions, average="macro")

    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names
    )

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "report": report,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def train_pipeline(
    train_cfg: TrainConfig, sentiment_cfg: SentimentConfig
) -> Dict[str, Any]:
    """
    Runs the full training pipeline: data loading, preprocessing,
    vocabulary creation, dataset/dataloader setup, model training, and evaluation.
    """
    print("\n--- Starting Training Pipeline ---")

    # 1. Data Loading and Preprocessing
    df_raw = download_and_load_data(train_cfg.data_url, train_cfg.local_raw_data_path)
    df_combined = combine_comments(df_raw.copy())
    preprocessor = DataPreprocessor()
    comment_df = extract_and_process_comments(df_combined.copy(), preprocessor)

    # Filter English comments and create ensemble sentiment
    english_comments = comment_df[comment_df["language"] == "en"].copy()
    english_comments["final_sentiment"] = english_comments.apply(
        create_ensemble_sentiment, axis=1
    )
    print(f"Filtered to {len(english_comments)} English comments.")
    print("Final sentiment distribution after ensemble:")
    print(english_comments["final_sentiment"].value_counts())

    # Ensure 'Post_ID' is handled if it's not in the original data or if it's needed for merging later
    if "Post_ID" not in english_comments.columns:
        english_comments["Post_ID"] = range(
            len(english_comments)
        )  # Dummy Post_ID if not present

    # 2. Data Splitting
    X = english_comments["cleaned_comment"].values
    y = english_comments["final_sentiment"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=train_cfg.val_size,
        random_state=train_cfg.random_state,
        stratify=y_temp,
    )

    print(
        f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} samples."
    )

    # 3. Vocabulary Creation
    vocab = Vocabulary(
        X_train.tolist(),
        min_freq=sentiment_cfg.min_freq,
        max_vocab_size=sentiment_cfg.max_vocab_size,
    )
    print(f"Vocabulary size: {len(vocab)}")

    # Determine max_seq_length from training data
    train_lengths = [len(vocab.encode(text)) for text in X_train]
    max_seq_length = int(np.percentile(train_lengths, 95))
    sentiment_cfg.max_seq_length = max_seq_length  # Update config
    print(f"Max sequence length (95th percentile of training data): {max_seq_length}")

    # Save vocabulary and label map
    with open(train_cfg.vocab_save_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {train_cfg.vocab_save_path}")

    label_map = {"negative": 0, "neutral": 1, "positive": 2}  # Fixed mapping
    with open(train_cfg.label_map_save_path, "w") as f:
        json.dump(label_map, f)
    print(f"Label map saved to {train_cfg.label_map_save_path}")

    # 4. PyTorch Dataset & DataLoader
    train_dataset = SentimentDataset(
        X_train.tolist(), y_train.tolist(), vocab, sentiment_cfg.max_seq_length
    )
    val_dataset = SentimentDataset(
        X_val.tolist(), y_val.tolist(), vocab, sentiment_cfg.max_seq_length
    )
    test_dataset = SentimentDataset(
        X_test.tolist(), y_test.tolist(), vocab, sentiment_cfg.max_seq_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"Data loaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    # 5. CNN Model Definition and Training
    cnn_model = CNNSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=sentiment_cfg.embed_dim,
        output_dim=sentiment_cfg.num_classes,
        filter_sizes=sentiment_cfg.filter_sizes,
        num_filters=sentiment_cfg.num_filters,
        dropout=sentiment_cfg.dropout,
    )
    print(
        f"CNN model initialized with {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad):,} trainable parameters."
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=sentiment_cfg.lr)

    trainer = ModelTrainer(cnn_model, device, train_loader, val_loader)
    train_losses, val_losses, val_accuracies = trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=train_cfg.num_epochs,
        patience=train_cfg.patience,
    )

    # 6. Evaluate on Test Set
    print("\n--- Evaluating CNN Model on Test Set ---")
    test_results = evaluate_model(cnn_model, test_loader, device, label_map)

    print(f"CNN Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"CNN Test F1-Weighted: {test_results['f1_weighted']:.4f}")
    print(f"CNN Test F1-Macro: {test_results['f1_macro']:.4f}")
    print("\nCNN Classification Report:\n", test_results["report"])

    # 7. Save Training Results
    training_summary = {
        "final_test_accuracy": test_results["accuracy"],
        "final_test_f1_weighted": test_results["f1_weighted"],
        "final_test_f1_macro": test_results["f1_macro"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "model_path": train_cfg.model_save_path,
        "vocab_path": train_cfg.vocab_save_path,
        "label_map_path": train_cfg.label_map_save_path,
        "train_config": train_cfg.__dict__,
        "sentiment_config": sentiment_cfg.__dict__,
    }
    with open(train_cfg.results_save_path, "w") as f:
        json.dump(training_summary, f, indent=4)
    print(f"Training results saved to {train_cfg.results_save_path}")

    print("\n--- Training Pipeline Complete ---")
    return training_summary


if __name__ == "__main__":
    # Example usage for testing trainer.py directly
    # Ensure 'Post_ID' column exists in your raw data or add a dummy one for testing
    # The GCS URL is hardcoded in TrainConfig for this example.
    train_config = TrainConfig(
        num_epochs=5, patience=2
    )  # Reduced epochs/patience for quick test
    sentiment_config = SentimentConfig()

    # Run the full training pipeline
    results = train_pipeline(train_config, sentiment_config)
    print("\nDirect run training summary:", results)
