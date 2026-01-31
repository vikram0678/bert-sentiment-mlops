import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

def batch_predict(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading model for batch prediction...")
    tokenizer = AutoTokenizer.from_pretrained("model_output")
    model = AutoModelForSequenceClassification.from_pretrained("model_output")
    
    df = pd.read_csv(input_path)
    if 'text' not in df.columns:
        print("Error: CSV must have a 'text' column.")
        return

    results = []
    confidences = []

    print(f"Processing {len(df)} rows...")
    for text in df['text']:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            results.append("positive" if pred.item() == 1 else "negative")
            confidences.append(float(conf))

    df['sentiment'] = results
    df['confidence'] = confidences
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Results saved to {output_path}")

if __name__ == "__main__":
    # Command: python scripts/batch_predict.py input.csv output.csv
    if len(sys.argv) < 3:
        print("Usage: python batch_predict.py <input_csv> <output_csv>")
    else:
        batch_predict(sys.argv[1], sys.argv[2])