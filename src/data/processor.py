import pandas as pd
import numpy as np
import re
import os
import warnings
from typing import List, Tuple, Dict, Any

# NLP and text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

warnings.filterwarnings("ignore")

# --- NLTK Downloads ---
# These downloads are crucial for the DataPreprocessor to work.
# They should ideally be handled during environment setup, but including here for robustness.
try:
    nltk.data.find('corpora/stopwords')
except Exception as e: # Changed from nltk.downloader.DownloadError
    print(f"Downloading NLTK stopwords: {e}") # Added print for debugging
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except Exception as e: # Changed from nltk.downloader.DownloadError
    print(f"Downloading NLTK wordnet: {e}") # Added print for debugging
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except Exception as e: # Changed from nltk.downloader.DownloadError
    print(f"Downloading NLTK punkt: {e}") # Added print for debugging
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('sentiment/vader_lexicon')
except Exception as e: # Changed from nltk.downloader.DownloadError
    print(f"Downloading NLTK vader_lexicon: {e}") # Added print for debugging
    nltk.download('vader_lexicon', quiet=True)


class DataPreprocessor:
    """
    Handles text cleaning, language detection, and rule-based sentiment analysis (VADER, TextBlob).
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove URLs, mentions, hashtags
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove special characters and digits but keep basic punctuation
        text = re.sub(r"[^a-zA-Z\s.,!?]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]  # Filter out very short words

        return " ".join(tokens)

    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if not text or text.strip() == "":
                return "unknown"
            return detect(text)
        except LangDetectException:
            return "unknown"

    def get_vader_sentiment(self, text: str) -> Tuple[str, float]:
        """Get sentiment using VADER"""
        if not text or text == "":
            return "neutral", 0.0

        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            return "positive", compound
        elif compound <= -0.05:
            return "negative", compound
        else:
            return "neutral", compound

    def get_textblob_sentiment(self, text: str) -> Tuple[str, float]:
        """Get sentiment using TextBlob"""
        if not text or text == "":
            return "neutral", 0.0

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:  # A slightly wider neutral range for TextBlob
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity


def download_and_load_data(
    gcs_url: str, local_path: str = "data/raw/social_media_data.csv"
) -> pd.DataFrame:
    """
    Downloads data from a GCS URL and loads it into a pandas DataFrame.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading data from {gcs_url} to {local_path}...")
    try:
        # Using pandas directly to read from URL is often simpler for CSVs
        df = pd.read_csv(gcs_url)
        df.to_csv(local_path, index=False)  # Save a local copy
        print("Data downloaded and loaded successfully.")
        return df
    except Exception as e:
        print(f"Error downloading or loading data: {e}")
        # Fallback to local if already exists
        if os.path.exists(local_path):
            print(f"Attempting to load from existing local file: {local_path}")
            return pd.read_csv(local_path)
        raise


def combine_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines 'Comment 1' through 'Comment 10' columns into a single 'combined_comments' column.
    """
    comment_cols = [col for col in df.columns if re.match(r"Comment \d+", col)]
    df["combined_comments"] = df[comment_cols].fillna("").agg(" ".join, axis=1)
    print("Comments combined into 'combined_comments' column.")
    return df


def extract_and_process_comments(
    df: pd.DataFrame, preprocessor: DataPreprocessor
) -> pd.DataFrame:
    """
    Extracts all comments from the dataframe, applies preprocessing,
    language detection, and rule-based sentiment analysis.
    """
    comment_data = []
    comment_cols = [f"Comment {i}" for i in range(1, 11)]
    existing_comment_cols = [col for col in comment_cols if col in df.columns]

    print("Processing comments...")
    for idx, row in df.iterrows():
        for col in existing_comment_cols:
            comment_text = row[col]
            if pd.notna(comment_text) and str(comment_text).strip() != "":
                original_comment = str(comment_text)
                cleaned_comment = preprocessor.clean_text(original_comment)

                if (
                    cleaned_comment and len(cleaned_comment.split()) >= 2
                ):  # Ensure cleaned comment is meaningful
                    language = preprocessor.detect_language(original_comment)
                    vader_sentiment, vader_score = preprocessor.get_vader_sentiment(
                        original_comment
                    )
                    textblob_sentiment, textblob_score = (
                        preprocessor.get_textblob_sentiment(original_comment)
                    )

                    comment_data.append(
                        {
                            "post_id": row[
                                "Post_ID"
                            ],  # Assuming 'Post_ID' exists in your raw data
                            "original_comment": original_comment,
                            "cleaned_comment": cleaned_comment,
                            "comment_length": len(original_comment.split()),
                            "cleaned_length": len(cleaned_comment.split()),
                            "language": language,
                            "vader_sentiment": vader_sentiment,
                            "vader_score": vader_score,
                            "textblob_sentiment": textblob_sentiment,
                            "textblob_score": textblob_score,
                        }
                    )
    processed_comment_df = pd.DataFrame(comment_data)
    print(f"Processed {len(processed_comment_df)} comments.")
    return processed_comment_df


def create_ensemble_sentiment(row: pd.Series) -> str:
    """
    Creates an ensemble sentiment label from VADER and TextBlob.
    """
    vader_sent = row["vader_sentiment"]
    textblob_sent = row["textblob_sentiment"]

    if vader_sent == textblob_sent:
        return vader_sent

    # If they disagree, use the one with stronger absolute score
    if abs(row["vader_score"]) > abs(row["textblob_score"]):
        return vader_sent
    else:
        return textblob_sent


def save_processed_data(
    df: pd.DataFrame, path: str = "data/processed/processed_social_media_data.csv"
):
    """
    Saves the processed DataFrame to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")


if __name__ == "__main__":
    # Example usage for testing processor.py directly
    print("Running processor.py as main for testing purposes.")
    GCS_DATA_URL = "https://storage.googleapis.com/sachin_data1/_Social%20Media%20Analytics%20-%20LLM%20-%20Socila%20Media%20Analytics.csv"
    LOCAL_RAW_DATA_PATH = "data/raw/social_media_data.csv"
    PROCESSED_DATA_PATH = "data/processed/processed_social_media_data.csv"

    # 1. Download and Load Data
    df_raw = download_and_load_data(GCS_DATA_URL, LOCAL_RAW_DATA_PATH)

    # 2. Combine Comments
    df_combined = combine_comments(df_raw.copy())

    # 3. Initialize Preprocessor and Extract/Process Comments
    preprocessor = DataPreprocessor()
    comment_df = extract_and_process_comments(df_combined.copy(), preprocessor)

    # 4. Filter English comments and create ensemble sentiment
    english_comments = comment_df[comment_df["language"] == "en"].copy()
    english_comments["final_sentiment"] = english_comments.apply(
        create_ensemble_sentiment, axis=1
    )
    print(f"Filtered to {len(english_comments)} English comments.")
    print("\nFinal sentiment distribution after ensemble:")
    print(english_comments["final_sentiment"].value_counts())

    # 5. Save processed data
    save_processed_data(english_comments, PROCESSED_DATA_PATH)
    print(f"\nProcessed data saved to {PROCESSED_DATA_PATH}")
    print("\nExample of processed data:")
    print(
        english_comments[
            ["original_comment", "cleaned_comment", "final_sentiment"]
        ].head()
    )
