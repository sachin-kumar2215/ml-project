import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Explore Data",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explore Your Dataset")
st.markdown("This page allows you to visualize and understand your processed data.")

PROCESSED_COMMENTS_PATH = 'data/processed/preprocessed_comments_for_pipeline.csv'

# Check if the processed data file exists
if os.path.exists(PROCESSED_COMMENTS_PATH):
    try:
        processed_df = pd.read_csv(PROCESSED_COMMENTS_PATH)
        st.success(f"Loaded {len(processed_df)} rows of processed data.")

        st.subheader("Processed Data Sample")
        st.dataframe(processed_df.head())

        st.subheader("Data Overview")
        st.write(f"Total comments processed: {len(processed_df)}")
        st.write(f"Columns: {processed_df.columns.tolist()}")

        # Display basic statistics
        st.subheader("Descriptive Statistics for Numerical Columns")
        st.dataframe(processed_df.describe())

        # Visualization 1: Language Distribution
        st.subheader("Language Distribution of Processed Comments")
        if 'language' in processed_df.columns:
            lang_counts = processed_df['language'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=lang_counts.index, y=lang_counts.values, ax=ax, palette='viridis')
            ax.set_title('Language Distribution')
            ax.set_xlabel('Language')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.info("Language column not found for plotting.")

        # Visualization 2: Final Sentiment Distribution
        st.subheader("Final Sentiment Distribution")
        if 'final_sentiment' in processed_df.columns:
            sentiment_counts = processed_df['final_sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette='coolwarm')
            ax.set_title('Final Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.info("Final sentiment column not found for plotting.")

        # Visualization 3: Comment Length Distribution (Cleaned)
        st.subheader("Cleaned Comment Length Distribution")
        if 'cleaned_length' in processed_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(processed_df['cleaned_length'], bins=50, kde=True, ax=ax, color='purple')
            ax.set_title('Cleaned Comment Length Distribution')
            ax.set_xlabel('Number of Words (Cleaned)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.info("Cleaned length column not found for plotting.")

    except Exception as e:
        st.error(f"An error occurred while loading or displaying data: {e}")
        st.exception(e) # Show full traceback for debugging
else:
    st.warning(f"Processed data file not found at `{PROCESSED_COMMENTS_PATH}`. Please run the ML pipeline training or prediction first to generate this file.")
    st.info("You can run the training pipeline using: `python -m src.main --mode train --train_sentiment --data_path \"<your_data_url>\"`")