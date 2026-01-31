# 🐦 BERT Sentiment Analysis MLOps Pipeline

A complete end-to-end MLOps solution for sentiment analysis on Twitter/IMDB data using a fine-tuned DistilBERT model. This project demonstrates a production-grade workflow including data preprocessing, memory-optimized training, and a containerized microservices architecture.

---

## 🏗️ Project Structure

```
bert-sentiment-mlops/
├── data/
│   └── processed/                    # Preprocessed CSVs (train.csv, test.csv)
├── model_output/                     # Saved model weights, config, and tokenizer
├── results/                          # Metrics, run summaries, and batch predictions
├── scripts/                          # Core ML Logic
│   ├── preprocess.py                # Data cleaning & train/test split
│   ├── train.py                     # Fine-tuning logic & experiment tracking
│   └── batch_predict.py             # Bulk inference on CSV files
├── src/                              # Application Services
│   ├── api.py                       # FastAPI backend
│   └── ui.py                        # Streamlit frontend
├── Dockerfile.api                    # Backend container environment
├── Dockerfile.ui                     # Frontend container environment
├── docker-compose.yml                # Multi-service orchestration
├── .env.example                      # Environment variable template
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Setup Environment

First, clone the repository and prepare your environment variables.

```bash
git clone https://github.com/vikram0678/bert-sentiment-mlops.git
cd bert-sentiment-mlops

# Create the .env file from the provided template
cp .env.example .env
```

### 2. Execution Workflow

For a clean and successful deployment, follow these three phases:

---

## 📋 Execution Phases

### Phase A: Training (The Intelligence)

This step runs the preprocessing and training scripts inside a container to generate the model artifacts and metrics.

```bash
docker-compose up trainer
```

**Note:** This generates `model_output/` and `results/metrics.json`.

---

### Phase B: Deployment (The Services)

Build and launch the API and UI services. Using the `--build` flag ensures your latest code changes are applied.

```bash
docker-compose up -d --build api ui
```

---

### Phase C: Validation

Verify that all services are running correctly:

**Web Interface:**
Access the UI at http://localhost:8501

**API Health:**
Check status at http://localhost:8000/health

**System Check:**
Run `docker ps` to verify both services are healthy.

---

## 📊 Batch Prediction

To process an entire file of text at once, run the batch script within the running API container:

```bash
docker compose exec api python scripts/batch_predict.py data/processed/small_test.csv results/predictions.csv
```

---

## 🧠 Model & Technical Choices

**Model:** `distilbert-base-uncased` — chosen for its high accuracy-to-speed ratio, making it ideal for CPU-based real-time inference.

**Optimization:** Utilized Gradient Accumulation (8 steps) during training to achieve stable convergence while maintaining a low memory footprint.

**Architecture:** Decoupled Backend (FastAPI) and Frontend (Streamlit) services communicating over a virtual Docker network.

---

## 🧪 Testing

To run the automated test suite for the API and Model loading:

```bash
docker compose exec api python tests/test_api.py
docker compose exec api python tests/test_model.py
```

---