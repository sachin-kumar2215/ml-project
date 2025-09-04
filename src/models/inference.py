import pandas as pd
import torch
import torch.nn as nn
import os
import json
import pickle # For loading Vocabulary
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Import your CNN model definition
from src.models.sentiment_model import CNNSentimentClassifier
# Import data preprocessing functions and classes
from src.data.processor import DataPreprocessor
# Import Vocabulary and SentimentDataset from trainer for consistency
from src.models.trainer import Vocabulary, SentimentDataset, collate_fn, SentimentConfig

# Set up device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class InferenceConfig:
    model_path: str = 'models/best_cnn_model.pth'
    vocab_path: str = 'models/vocabulary.pkl'
    label_map_path: str = 'models/label_map.json'
    batch_size: int = 64 # Batch size for inference

def load_sentiment_artifacts(config: InferenceConfig) -> Dict[str, Any]:
    """
    Loads the trained PyTorch CNN model, Vocabulary, and label map.
    """
    print(f"Loading vocabulary from {config.vocab_path}...")
    if not os.path.exists(config.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {config.vocab_path}. Please train the model first.")
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(f"Loading label map from {config.label_map_path}...")
    if not os.path.exists(config.label_map_path):
        raise FileNotFoundError(f"Label map file not found at {config.label_map_path}. Please train the model first.")
    with open(config.label_map_path, 'r') as f:
        label_map = json.load(f)
    reverse_label_map = {v: k for k, v in label_map.items()}

    print(f"Loading CNN model from {config.model_path}...")
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model file not found at {config.model_path}. Please train the model first.")

    # We need to know the model's architecture parameters (vocab_size, embed_dim, etc.)
    # These should ideally be saved with the model or derived from the vocab.
    # For now, we'll use default SentimentConfig values and vocab size.
    # A more robust solution would save these params during training.
    sentiment_cfg = SentimentConfig() # Load default config
    sentiment_cfg.vocab_size = len(vocab) # Update vocab size from loaded vocab

    # Instantiate the model with the correct architecture
    model = CNNSentimentClassifier(
        vocab_size=sentiment_cfg.vocab_size,
        embed_dim=sentiment_cfg.embed_dim,
        output_dim=sentiment_cfg.num_classes,
        filter_sizes=sentiment_cfg.filter_sizes,
        num_filters=sentiment_cfg.num_filters,
        dropout=sentiment_cfg.dropout
    ).to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.eval() # Set model to evaluation mode

    print("Model artifacts loaded successfully.")
    return {
        "model": model,
        "vocab": vocab,
        "label_map": label_map,
        "reverse_label_map": reverse_label_map,
        "sentiment_config": sentiment_cfg # Pass the config for max_seq_length
    }

def predict_sentiment(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Performs sentiment prediction on new comment data and integrates scores.
    Assumes df has 'combined_comments' column.
    """
    model = artifacts["model"]
    vocab = artifacts["vocab"]
    reverse_label_map = artifacts["reverse_label_map"]
    sentiment_cfg = artifacts["sentiment_config"] # Contains max_seq_length

    preprocessor = DataPreprocessor() # Initialize preprocessor for cleaning new text

    print("Preprocessing text for inference...")
    # Apply the same cleaning as during training
    df['cleaned_comments'] = df['combined_comments'].apply(preprocessor.clean_text)

    # Filter out empty cleaned comments to avoid errors in encoding
    df_filtered = df[df['cleaned_comments'].str.strip() != ''].copy()
    if df_filtered.empty:
        print("No valid comments to predict after cleaning.")
        df['Overall_Comment_Sentiment'] = 'neutral' # Default or handle as needed
        df['Sentiment_Prob_negative'] = 0.0
        df['Sentiment_Prob_neutral'] = 0.0
        df['Sentiment_Prob_positive'] = 0.0
        return df

    # Create a temporary Dataset and DataLoader for inference
    # Labels are dummy here as we don't have them for inference
    inference_dataset = SentimentDataset(
        df_filtered['cleaned_comments'].tolist(),
        ['neutral'] * len(df_filtered), # Dummy labels
        vocab,
        sentiment_cfg.max_seq_length # Use max_seq_length from training
    )
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=InferenceConfig().batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_predictions = []
    all_probabilities = []

    print("Making sentiment predictions...")
    with torch.no_grad():
        for batch in inference_loader:
            texts = batch['texts'].to(device)
            lengths = batch['lengths']

            outputs = model(texts, lengths)
            probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Map numerical predictions back to original labels
    predicted_labels = [reverse_label_map[p] for p in all_predictions]

    # Add predictions and probabilities back to the filtered DataFrame
    df_filtered['Overall_Comment_Sentiment'] = predicted_labels
    prob_df = pd.DataFrame(all_probabilities, columns=[f'Sentiment_Prob_{reverse_label_map[i]}' for i in range(sentiment_cfg.num_classes)])
    df_filtered = pd.concat([df_filtered.reset_index(drop=True), prob_df], axis=1)

    # Merge predictions back to the original DataFrame, handling comments that were filtered out
    # This ensures the output DataFrame has the same number of rows as the input
    df_output = df.copy()
    df_output = df_output.merge(df_filtered[['original_comment', 'Overall_Comment_Sentiment', 'Sentiment_Prob_negative', 'Sentiment_Prob_neutral', 'Sentiment_Prob_positive']],
                                on='original_comment', how='left', suffixes=('', '_predicted'))

    # Fill NaN predictions for comments that were empty/invalid after cleaning
    df_output['Overall_Comment_Sentiment'].fillna('neutral', inplace=True) # Default for unpredicted
    for col in ['Sentiment_Prob_negative', 'Sentiment_Prob_neutral', 'Sentiment_Prob_positive']:
        df_output[col].fillna(0.0, inplace=True) # Default probabilities

    print("Sentiment prediction complete.")
    return df_output

if __name__ == "__main__":
    # Example usage for testing inference.py directly
    print("Running inference.py as main for testing purposes.")

    # Ensure models directory exists and contains artifacts from training
    # For this test, you MUST have run the training pipeline at least once
    # (e.g., via src/models/trainer.py or src/main.py)
    if not os.path.exists('models/best_cnn_model.pth') or \
       not os.path.exists('models/vocabulary.pkl') or \
       not os.path.exists('models/label_map.json'):
        print("Error: Model artifacts not found. Please run trainer.py or main.py in 'train' mode first.")
        exit()

    inference_cfg = InferenceConfig()
    artifacts = load_sentiment_artifacts(inference_cfg)

    # Create dummy data for inference
    dummy_inference_data = {
        'Comment 1': ["This is an amazing product!", "I am so disappointed with this.", "It's just average.", "Absolutely fantastic!", "Worst purchase ever."],
        'Comment 2': ["Highly recommend.", "Never again.", "Could be better.", "A must-buy!", "Total waste of money."],
        'Post_ID': [1, 2, 3, 4, 5] # Dummy Post_ID
    }
    dummy_df_inference = pd.DataFrame(dummy_inference_data)
    dummy_df_inference['combined_comments'] = dummy_df_inference[['Comment 1', 'Comment 2']].fillna('').agg(' '.join, axis=1)
    dummy_df_inference['original_comment'] = dummy_df_inference['combined_comments'] # For merging back

    df_with_sentiment = predict_sentiment(dummy_df_inference, artifacts)

    print("\nInference Results:")
    print(df_with_sentiment[['original_comment', 'Overall_Comment_Sentiment', 'Sentiment_Prob_positive', 'Sentiment_Prob_neutral', 'Sentiment_Prob_negative']].head())