from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Dict, Any
import sys

# Import necessary functions from your ML pipeline
from src.models.inference import load_sentiment_artifacts, predict_sentiment, InferenceConfig
from src.data.processor import DataPreprocessor # For internal preprocessing if needed

app = FastAPI(
    title="Social Media Sentiment Analysis API",
    description="API for predicting sentiment of social media comments using a PyTorch CNN model.",
    version="1.0.0"
)

# Global variable to store loaded model artifacts
model_artifacts: Dict[str, Any] = {}

# Pydantic model for request body
class PredictionRequest(BaseModel):
    text: List[str] # Accepts a list of strings for batch prediction

# Pydantic model for response body
class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]

@app.on_event("startup")
async def load_model_on_startup():
    """
    Load the model and other artifacts when the FastAPI application starts.
    """
    global model_artifacts
    print("Loading model artifacts on startup...")
    try:
        inference_config = InferenceConfig()
        model_artifacts = load_sentiment_artifacts(inference_config)
        print("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model artifacts: {e}. Please ensure models are trained and saved.")
        # Raise HTTPException to prevent the app from starting without a model
        raise HTTPException(status_code=500, detail=f"Model artifacts not found: {e}. Run training first.")
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Social Media Sentiment Analysis API. Visit /docs for API documentation."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts sentiment for a list of input texts.
    """
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is still starting or encountered an error.")

    try:
        # Create a DataFrame from the input texts
        # The predict_sentiment function expects 'combined_comments'
        input_df = pd.DataFrame({'combined_comments': request.text})

        # Perform prediction using the loaded artifacts
        df_with_sentiment = predict_sentiment(input_df, model_artifacts)

        # Prepare response
        # Convert relevant columns to dictionary for JSON response
        results = df_with_sentiment[['combined_comments', 'Overall_Comment_Sentiment',
                                     'Sentiment_Prob_negative', 'Sentiment_Prob_neutral',
                                     'Sentiment_Prob_positive']].to_dict(orient='records')

        return PredictionResponse(predictions=results)

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Ensure models directory exists and contains artifacts for local testing
    if not os.path.exists('models/best_cnn_model.pth') or \
       not os.path.exists('models/vocabulary.pkl') or \
       not os.path.exists('models/label_map.json'):
        print("Warning: Model artifacts not found. Please run training (python -m src.main --mode train --train_sentiment ...) first.")
        print("API will attempt to start, but prediction endpoints will fail until model is trained.")
        # sys.exit(1) # Uncomment to force exit if model not found

    uvicorn.run(app, host="0.0.0.0", port=8000)