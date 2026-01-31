import requests

def test_health_endpoint():
    # Test the health check
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_endpoint():
    # Test a real prediction
    payload = {"text": "I love this project!"}
    response = requests.post("http://localhost:8000/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data

if __name__ == "__main__":
    print("Running API tests...")
    try:
        test_health_endpoint()
        test_prediction_endpoint()
        print("✅ API Tests Passed!")
    except Exception as e:
        print(f"❌ API Tests Failed: {e}")