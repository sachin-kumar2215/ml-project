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

    # Create a copy of the input DataFrame and add a unique ID for merging
    df_with_predictions = df.copy()
    df_with_predictions['__temp_id__'] = range(len(df_with_predictions))

    # Apply the same cleaning as during training
    df_with_predictions['cleaned_comments'] = df_with_predictions['combined_comments'].apply(preprocessor.clean_text)

    # Filter out empty cleaned comments for prediction
    df_to_predict = df_with_predictions[df_with_predictions['cleaned_comments'].str.strip() != ''].copy()

    if df_to_predict.empty:
        print("No valid comments to predict after cleaning. Returning neutral predictions.")
        df_with_predictions['Overall_Comment_Sentiment'] = 'neutral'
        df_with_predictions['Sentiment_Prob_negative'] = 0.0
        df_with_predictions['Sentiment_Prob_neutral'] = 1.0 # Default to neutral with 1.0 prob
        df_with_predictions['Sentiment_Prob_positive'] = 0.0
        return df_with_predictions.drop(columns=['__temp_id__', 'cleaned_comments'])


    # Create a temporary Dataset and DataLoader for inference
    inference_dataset = SentimentDataset(
        df_to_predict['cleaned_comments'].tolist(),
        ['neutral'] * len(df_to_predict), # Dummy labels
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

    # Add predictions and probabilities back to the DataFrame that was actually predicted
    df_to_predict['Overall_Comment_Sentiment'] = predicted_labels
    prob_df = pd.DataFrame(all_probabilities, columns=[f'Sentiment_Prob_{reverse_label_map[i]}' for i in range(sentiment_cfg.num_classes)])
    df_to_predict = pd.concat([df_to_predict.reset_index(drop=True), prob_df], axis=1)

    # Merge predictions back to the original DataFrame using the temporary ID
    df_final = df_with_predictions.merge(
        df_to_predict[['__temp_id__', 'Overall_Comment_Sentiment', 'Sentiment_Prob_negative', 'Sentiment_Prob_neutral', 'Sentiment_Prob_positive']],
        on='__temp_id__',
        how='left'
    )

    # Fill NaN predictions for comments that were empty/invalid after cleaning
    df_final['Overall_Comment_Sentiment'].fillna('neutral', inplace=True) # Default for unpredicted
    df_final['Sentiment_Prob_negative'].fillna(0.0, inplace=True)
    df_final['Sentiment_Prob_neutral'].fillna(1.0, inplace=True) # Default to neutral with 1.0 prob
    df_final['Sentiment_Prob_positive'].fillna(0.0, inplace=True)

    # Clean up temporary columns
    df_final.drop(columns=['__temp_id__', 'cleaned_comments'], inplace=True)

    print("Sentiment prediction complete.")
    return df_final