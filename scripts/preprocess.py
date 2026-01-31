import pandas as pd
from datasets import load_dataset
import os
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def preprocess():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Clean the text
    print("Cleaning text...")
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    
    # Ensure directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save as CSV (Mandatory Requirement)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print("Preprocessing complete. Files saved to data/processed/")

if __name__ == "__main__":
    preprocess()