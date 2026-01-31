# BERT Sentiment MLOps Pipeline

A containerized sentiment analysis service using a fine-tuned DistilBERT model.

## 🚀 How to Run
1. **Initialize Environment:**
   `cp env.example .env`
2. **Build and Run:**
   `docker-compose up --build`
   *This will automatically preprocess data, train the model, and launch the services.*

## 🌐 Services
- **API:** http://localhost:8000 (FastAPI)
- **Web UI:** http://localhost:8501 (Streamlit)

## 📊 Batch Prediction
Run the following inside the container:
`docker compose exec api python scripts/batch_predict.py data/processed/test.csv results/predictions.csv`