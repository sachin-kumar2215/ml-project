import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Configure plotting (can be done once at the start of your main script or here)
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)


def plot_initial_data_distributions(df: pd.DataFrame):
    """
    Generates and displays initial data distribution plots from the raw DataFrame.
    Assumes 'Platform', 'Sentiment_Score', 'Engagement_Rate', 'Number_of_Likes' columns exist.
    """
    print("\n--- Generating Initial Data Distributions ---")
    fig, axes = plt.subplots(2, 3, figsize=(25, 15)) # Adjusted size for better visibility
    fig.suptitle('Initial Data Distributions', fontsize=20)

    # Platform Distribution
    df['Platform'].value_counts().plot(kind='bar', ax=axes[0,0], color=sns.color_palette("viridis", len(df['Platform'].unique())))
    axes[0,0].set_title('Platform Distribution', fontsize=16)
    axes[0,0].set_xlabel('Platform', fontsize=14)
    axes[0,0].set_ylabel('Number of Posts', fontsize=14)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Original Sentiment Score Distribution
    if 'Sentiment_Score' in df.columns:
        df['Sentiment_Score'].value_counts().plot(kind='bar', ax=axes[0,1], color=sns.color_palette("plasma", len(df['Sentiment_Score'].unique())))
        axes[0,1].set_title('Original Sentiment Score Distribution', fontsize=16)
        axes[0,1].set_xlabel('Sentiment Score', fontsize=14)
        axes[0,1].set_ylabel('Number of Posts', fontsize=14)
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].set_title('Sentiment_Score Column Not Found', fontsize=16)
        axes[0,1].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[0,1].transAxes, fontsize=20)


    # Comments per Post Distribution (requires 'total_comments' column)
    if 'total_comments' in df.columns:
        df['total_comments'].value_counts().sort_index().plot(kind='bar', ax=axes[0,2], color=sns.color_palette("magma", len(df['total_comments'].unique())))
        axes[0,2].set_title('Number of Comments per Post', fontsize=16)
        axes[0,2].set_xlabel('Number of Comments', fontsize=14)
        axes[0,2].set_ylabel('Number of Posts', fontsize=14)
        axes[0,2].tick_params(axis='x', rotation=0)
    else:
        axes[0,2].set_title('total_comments Column Not Found', fontsize=16)
        axes[0,2].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[0,2].transAxes, fontsize=20)


    # Engagement Rate Distribution
    if 'Engagement_Rate' in df.columns:
        sns.histplot(df['Engagement_Rate'], bins=30, kde=True, ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('Engagement Rate Distribution', fontsize=16)
        axes[1,0].set_xlabel('Engagement Rate', fontsize=14)
        axes[1,0].set_ylabel('Frequency', fontsize=14)
    else:
        axes[1,0].set_title('Engagement_Rate Column Not Found', fontsize=16)
        axes[1,0].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[1,0].transAxes, fontsize=20)


    # Number of Likes Distribution
    if 'Number_of_Likes' in df.columns:
        sns.histplot(df['Number_of_Likes'], bins=30, kde=True, ax=axes[1,1], color='lightcoral')
        axes[1,1].set_title('Number of Likes Distribution', fontsize=16)
        axes[1,1].set_xlabel('Number of Likes', fontsize=14)
        axes[1,1].set_ylabel('Frequency', fontsize=14)
    else:
        axes[1,1].set_title('Number_of_Likes Column Not Found', fontsize=16)
        axes[1,1].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[1,1].transAxes, fontsize=20)


    # Engagement Rate vs. Sentiment Score (Box Plot)
    if 'Sentiment_Score' in df.columns and 'Engagement_Rate' in df.columns:
        sns.boxplot(x='Sentiment_Score', y='Engagement_Rate', data=df, ax=axes[1,2], palette='viridis')
        axes[1,2].set_title('Engagement Rate by Original Sentiment Score', fontsize=16)
        axes[1,2].set_xlabel('Sentiment Score', fontsize=14)
        axes[1,2].set_ylabel('Engagement Rate', fontsize=14)
    else:
        axes[1,2].set_title('Engagement_Rate or Sentiment_Score Column Not Found', fontsize=16)
        axes[1,2].text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=axes[1,2].transAxes, fontsize=20)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


def plot_preprocessing_insights(comment_df: pd.DataFrame, english_comments: pd.DataFrame):
    """
    Generates and displays plots related to text preprocessing insights.
    Assumes 'language', 'vader_sentiment', 'textblob_sentiment', 'comment_length',
    'cleaned_length', and 'final_sentiment' columns exist in the respective DataFrames.
    """
    print("\n--- Generating Preprocessing Insights Visualizations ---")

    # Language Distribution
    plt.figure(figsize=(12, 7))
    comment_df['language'].value_counts().plot(kind='bar', color=sns.color_palette("tab10"))
    plt.title('Language Distribution of All Comments', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Number of Comments', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sentiment Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    fig.suptitle('Sentiment Distribution Comparison', fontsize=20)

    comment_df['vader_sentiment'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('VADER Sentiment', fontsize=16)
    axes[0].set_xlabel('Sentiment', fontsize=14)
    axes[0].set_ylabel('Number of Comments', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)

    comment_df['textblob_sentiment'].value_counts().plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('TextBlob Sentiment', fontsize=16)
    axes[1].set_xlabel('Sentiment', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)

    english_comments['final_sentiment'].value_counts().plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Ensemble Final Sentiment (English Comments)', fontsize=16)
    axes[2].set_xlabel('Sentiment', fontsize=14)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Comment Length Distribution (Original vs. Cleaned)
    plt.figure(figsize=(12, 7))
    sns.histplot(english_comments['comment_length'], bins=50, color='blue', label='Original Length', kde=True, stat='density', alpha=0.5)
    sns.histplot(english_comments['cleaned_length'], bins=50, color='red', label='Cleaned Length', kde=True, stat='density', alpha=0.5)
    plt.title('Distribution of Comment Lengths (English Comments)', fontsize=16)
    plt.xlabel('Number of Words', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.xlim(0, 100) # Limit x-axis for better visualization of common lengths
    plt.tight_layout()
    plt.show()


def plot_data_split_verification(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
    """
    Generates and displays plots verifying sentiment distribution across data splits.
    """
    print("\n--- Generating Data Split Verification Visualizations ---")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    fig.suptitle('Sentiment Distribution Across Data Splits', fontsize=20)

    # Train set distribution
    pd.Series(y_train).value_counts(normalize=True).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Train Set', fontsize=16)
    axes[0].set_xlabel('Sentiment', fontsize=14)
    axes[0].set_ylabel('Proportion', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)

    # Validation set distribution
    pd.Series(y_val).value_counts(normalize=True).plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Validation Set', fontsize=16)
    axes[1].set_xlabel('Sentiment', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)

    # Test set distribution
    pd.Series(y_test).value_counts(normalize=True).plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Test Set', fontsize=16)
    axes[2].set_xlabel('Sentiment', fontsize=14)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float], val_accuracies: List[float], model_name: str = "CNN"):
    """
    Plots training and validation loss and accuracy curves.
    """
    print("\n--- Generating Training Curves ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=16)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title(f'{model_name} - Validation Accuracy', fontsize=16)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels: List[int], predictions: List[int], label_map: Dict[str, int], title: str = 'Confusion Matrix'):
    """
    Plots a confusion matrix.
    """
    print("\n--- Generating Confusion Matrix ---")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage for testing visualization functions
    print("Running eda_util.py as main for testing purposes.")

    # Dummy data for demonstration
    dummy_raw_data = pd.DataFrame({
        'Platform': ['Facebook', 'Twitter', 'Facebook', 'Instagram'],
        'Sentiment_Score': [1, -1, 0, 1],
        'Engagement_Rate': [0.05, 0.02, 0.08, 0.1],
        'Number_of_Likes': [100, 50, 120, 200],
        'Comment 1': ['Great post!', 'Bad content.', 'Neutral comment.', 'Awesome!'],
        'Comment 2': ['Loved it.', np.nan, np.nan, 'Super!']
    })
    dummy_raw_data['total_comments'] = dummy_raw_data[['Comment 1', 'Comment 2']].notna().sum(axis=1)

    plot_initial_data_distributions(dummy_raw_data)

    dummy_comment_data = pd.DataFrame({
        'language': ['en', 'es', 'en', 'en', 'fr'],
        'vader_sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral'],
        'textblob_sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral'],
        'comment_length': [10, 8, 12, 15, 7],
        'cleaned_length': [5, 4, 6, 7, 3],
        'final_sentiment': ['positive', 'negative', 'neutral', 'positive', 'neutral']
    })
    dummy_english_comments = dummy_comment_data[dummy_comment_data['language'] == 'en']

    plot_preprocessing_insights(dummy_comment_data, dummy_english_comments)

    dummy_y_train = np.array(['positive', 'negative', 'neutral', 'positive'])
    dummy_y_val = np.array(['negative', 'neutral'])
    dummy_y_test = np.array(['positive', 'negative'])

    plot_data_split_verification(dummy_y_train, dummy_y_val, dummy_y_test)

    dummy_train_losses = [0.5, 0.3, 0.1]
    dummy_val_losses = [0.6, 0.4, 0.2]
    dummy_val_accuracies = [70, 80, 90]

    plot_training_curves(dummy_train_losses, dummy_val_losses, dummy_val_accuracies)

    dummy_labels = [0, 1, 2, 0, 1, 2]
    dummy_predictions = [0, 1, 1, 0, 1, 2]
    dummy_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    plot_confusion_matrix(dummy_labels, dummy_predictions, dummy_label_map, title='Dummy Confusion Matrix')