import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_model_files():
    # Check if weights were saved correctly
    assert os.path.exists("model_output/config.json")
    assert os.path.exists("model_output/model.safetensors") or os.path.exists("model_output/pytorch_model.bin")

def test_model_loading():
    # Check if the model can be loaded into memory
    tokenizer = AutoTokenizer.from_pretrained("model_output")
    model = AutoModelForSequenceClassification.from_pretrained("model_output")
    assert model is not None
    print("✅ Model Loading Test Passed!")

if __name__ == "__main__":
    print("Running Model tests...")
    test_model_files()
    test_model_loading()