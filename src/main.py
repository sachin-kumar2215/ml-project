import argparse
import pandas as pd
import os
import json
from datetime import datetime
import sys

# Import functions and classes from your pipeline modules
from src.data.processor import download_and_load_data, combine_comments, DataPreprocessor, extract_and_process_comments, create_ensemble_sentiment, save_processed_data
from src.models.trainer import train_pipeline, TrainConfig, SentimentConfig
from src.models.inference import load_sentiment_artifacts, predict_sentiment, InferenceConfig
from src.visualization import eda_util # NEW IMPORT

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline for Social Media Analytics.")

    # --- General Arguments ---
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input CSV data (can be GCS URL or local path).")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], required=True,
                        help="Mode of operation: 'train' for model training, 'predict' for inference.")
    parser.add_argument("--output_path", type=str, default="data/processed/output_with_sentiment.csv",
                        help="Path to save the output CSV with sentiment predictions (only for 'predict' mode).")

    # --- Training Specific Arguments ---
    parser.add_argument("--train_sentiment", action="store_true",
                        help="Flag to train the sentiment analysis model.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs for model training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for model training.")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience for model training.")
    parser.add_argument("--model_save_path", type=str, default="models/best_cnn_model.pth",
                        help="Path to save the trained sentiment model.")
    parser.add_argument("--vocab_save_path", type=str, default="models/vocabulary.pkl",
                        help="Path to save the vocabulary pickle file.")
    parser.add_argument("--label_map_save_path", type=str, default="models/label_map.json",
                        help="Path to save the label map JSON file.")
    parser.add_argument("--results_save_path", type=str, default="models/training_results.json",
                        help="Path to save training results JSON file.")

    # --- Inference Specific Arguments ---
    parser.add_argument("--predict_sentiment", action="store_true",
                        help="Flag to perform sentiment prediction.")

    # --- Visualization Arguments (NEW) ---
    parser.add_argument("--plot_eda", action="store_true",
                        help="Generate and display initial EDA plots.")
    parser.add_argument("--plot_preprocessing", action="store_true",
                        help="Generate and display preprocessing insights plots.")
    parser.add_argument("--plot_data_split", action="store_true",
                        help="Generate and display data split verification plots (only in train mode).")
    parser.add_argument("--plot_training_curves", action="store_true",
                        help="Generate and display training loss/accuracy curves (only in train mode).")
    parser.add_argument("--plot_confusion_matrix", action="store_true",
                        help="Generate and display confusion matrix (only in train mode).")


    args = parser.parse_args()

    # Ensure necessary directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    try:
        # --- Data Loading and Initial Preprocessing (Common to both modes) ---
        print(f"Loading data from: {args.data_path}")
        if args.data_path.startswith("gs://") or args.data_path.startswith("http://") or args.data_path.startswith("https://"):
            df_raw = download_and_load_data(args.data_path, local_path='data/raw/downloaded_data.csv')
        else:
            df_raw = pd.read_csv(args.data_path)
            print(f"Loaded data from local path: {args.data_path}")

        # Add 'total_comments' for initial EDA plot if not already present (NEW)
        comment_cols_for_eda = [f'Comment {i}' for i in range(1, 11)]
        existing_comment_cols_for_eda = [col for col in comment_cols_for_eda if col in df_raw.columns]
        df_raw['total_comments'] = df_raw[existing_comment_cols_for_eda].notna().sum(axis=1)

        if args.plot_eda: # NEW
            eda_util.plot_initial_data_distributions(df_raw)


        # Combine comments and extract/process them
        df_combined = combine_comments(df_raw.copy())
        preprocessor = DataPreprocessor()
        comment_df = extract_and_process_comments(df_combined.copy(), preprocessor)

        # Filter English comments and create ensemble sentiment
        # This processed data will be used for both training and inference
        english_comments = comment_df[comment_df['language'] == 'en'].copy()
        english_comments['final_sentiment'] = english_comments.apply(create_ensemble_sentiment, axis=1)
        print(f"Filtered to {len(english_comments)} English comments for pipeline.")

        if args.plot_preprocessing: # NEW
            eda_util.plot_preprocessing_insights(comment_df, english_comments)


        # Ensure 'Post_ID' exists for consistency, especially if original data doesn't have it
        if 'Post_ID' not in english_comments.columns:
            english_comments['Post_ID'] = range(len(english_comments)) # Dummy Post_ID if not present

        # Save the preprocessed data that will be used by the pipeline
        save_processed_data(english_comments, 'data/processed/preprocessed_comments_for_pipeline.csv')
        print("Preprocessed comments saved to data/processed/preprocessed_comments_for_pipeline.csv")

        if args.mode == 'train':
            if args.train_sentiment:
                print("\n--- Training Sentiment Model ---")
                train_config = TrainConfig(
                    data_url=args.data_path, # Pass original data path for re-download if needed
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    model_save_path=args.model_save_path,
                    vocab_save_path=args.vocab_save_path,
                    label_map_save_path=args.label_map_save_path,
                    results_save_path=args.results_save_path
                )
                sentiment_config = SentimentConfig() # Uses default CNN params

                # The train_pipeline function handles all data splitting, vocab creation,
                # model instantiation, training, and saving artifacts.
                training_summary = train_pipeline(train_config, sentiment_config)

                # Extract data for plotting from training_summary (NEW)
                X_train_for_plot = training_summary['X_train_cleaned_comments']
                y_train_for_plot = training_summary['y_train_sentiment']
                X_val_for_plot = training_summary['X_val_cleaned_comments']
                y_val_for_plot = training_summary['y_val_sentiment']
                X_test_for_plot = training_summary['X_test_cleaned_comments']
                y_test_for_plot = training_summary['y_test_sentiment']
                label_map_for_plot = training_summary['label_map']


                print("\nSentiment Model Training Complete.")
                print(f"Model saved to: {training_summary['model_path']}")
                print(f"Vocabulary saved to: {training_summary['vocab_path']}")
                print(f"Label Map saved to: {training_summary['label_map_path']}")
                print(f"Final Test Accuracy: {training_summary['final_test_accuracy']:.4f}")

                # Plotting after training (NEW)
                if args.plot_data_split:
                    eda_util.plot_data_split_verification(y_train_for_plot, y_val_for_plot, y_test_for_plot)

                if args.plot_training_curves:
                    eda_util.plot_training_curves(
                        training_summary['train_losses'],
                        training_summary['val_losses'],
                        training_summary['val_accuracies']
                    )

                if args.plot_confusion_matrix:
                    eda_util.plot_confusion_matrix(
                        training_summary['labels'], # True labels from test set
                        training_summary['predictions'], # Predicted labels from test set
                        label_map_for_plot,
                        title='CNN Test Set Confusion Matrix'
                    )

            else:
                print("No specific training task specified. Use --train_sentiment.")

        elif args.mode == 'predict':
            if args.predict_sentiment:
                print("\n--- Performing Sentiment Prediction ---")
                inference_config = InferenceConfig(
                    model_path=args.model_save_path,
                    vocab_path=args.vocab_save_path,
                    label_map_path=args.label_map_save_path,
                    batch_size=args.batch_size
                )
                artifacts = load_sentiment_artifacts(inference_config)

                # The predict_sentiment function expects 'combined_comments'
                # We need to ensure the input DataFrame for prediction has this column.
                # For this main.py, we've already processed the data into 'english_comments'
                # which has 'cleaned_comment' and 'original_comment'.
                # The inference function expects 'combined_comments' as the raw input.
                # Let's use the 'original_comment' from the processed data for prediction.
                # We'll create a temporary DataFrame with 'combined_comments' for the inference function.
                df_for_inference = english_comments[['original_comment']].copy()
                df_for_inference.rename(columns={'original_comment': 'combined_comments'}, inplace=True)

                df_with_sentiment = predict_sentiment(df_for_inference, artifacts)

                # Merge the predictions back to the original full dataframe if needed,
                # or just save the df_with_sentiment which contains original_comment and predictions.
                # For simplicity, we'll save the df_with_sentiment directly.
                save_processed_data(df_with_sentiment, args.output_path)
                print(f"Sentiment predictions saved to: {args.output_path}")
                print("\nExample predictions:")
                print(df_with_sentiment[['combined_comments', 'Overall_Comment_Sentiment']].head())
            else:
                print("No specific prediction task specified. Use --predict_sentiment.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure model artifacts exist if in 'predict' mode, or data path is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()