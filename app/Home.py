import streamlit as st
import pandas as pd
import requests
import json
import io

# Configuration for the FastAPI backend
# IMPORTANT: If running Streamlit and FastAPI on the same VM, use localhost.
# If running Dockerized, this might be http://backend:8000 or similar internal Docker network name.
# If accessing from outside VM, use http://<YOUR_VM_IP>:8000
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Social Media Sentiment Analyzer")
st.markdown("Upload your social media comments data (CSV or JSON) to get sentiment predictions.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

data_df = None
if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    st.write(file_details)

    if uploaded_file.type == "text/csv":
        data_df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/json":
        data_df = pd.read_json(uploaded_file)

    if data_df is not None:
        st.subheader("Uploaded Data Preview")
        st.dataframe(data_df.head())

        # --- Model Selection (Placeholder for multiple models) ---
        st.subheader("Select Model for Prediction")
        model_options = ["Sentiment Analysis (CNN)"] # For now, only one model
        selected_model = st.selectbox("Choose a model:", model_options)

        # --- Prediction Button ---
        if st.button("Get Predictions"):
            # Check for 'Comment 1' or similar columns
            comment_cols = [col for col in data_df.columns if col.startswith('Comment ')]
            if not comment_cols:
                st.error("Error: No 'Comment' columns found (e.g., 'Comment 1', 'Comment 2'). Please ensure your data has comment columns.")
            else:
                with st.spinner("Getting predictions..."):
                    # Combine comments for sending to API
                    # Fill NaN with empty string before combining to avoid errors
                    data_df['combined_comments_for_api'] = data_df[comment_cols].fillna('').agg(' '.join, axis=1)
                    texts_to_predict = data_df['combined_comments_for_api'].tolist()

                    # Call FastAPI
                    try:
                        response = requests.post(f"{FASTAPI_URL}/predict", json={"text": texts_to_predict})
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        predictions_data = response.json()['predictions']

                        # Convert predictions to DataFrame and merge
                        predictions_df = pd.DataFrame(predictions_data)
                        
                        # Ensure original index is preserved if merging
                        # For simplicity, let's just display the predictions alongside original comments
                        # We need to align by the original text if possible, or by index if order is guaranteed
                        # The API returns 'combined_comments' which can be used for merging
                        
                        # Create a temporary DataFrame for merging with original data
                        # Use a unique identifier if available, otherwise rely on order
                        temp_df_for_merge = data_df.copy()
                        temp_df_for_merge['__temp_idx__'] = range(len(temp_df_for_merge))

                        # Merge predictions back to the original DataFrame
                        # Assuming the order of predictions matches the order of texts sent
                        result_df = pd.concat([temp_df_for_merge, predictions_df], axis=1)
                        
                        # Drop the temporary column
                        result_df.drop(columns=['__temp_idx__', 'combined_comments_for_api'], inplace=True)

                        st.subheader("Prediction Results")
                        st.dataframe(result_df)

                        # Option to download results
                        csv_output = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_output,
                            file_name="sentiment_predictions.csv",
                            mime="text/csv",
                        )

                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure it is running.")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API returned an error: {e.response.status_code} - {e.response.text}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.exception(e) # Display full exception for debugging

st.markdown("---")
st.markdown("### Explore Your Clean Dataset")
st.markdown("Use the sidebar to navigate to data exploration pages (coming soon!).")
# This is where you'd link to pages in app/pages/