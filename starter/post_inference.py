import requests

# Replace with your deployed FastAPI URL
API_URL = "https://nd0821-c3-starter-code-scalable-ml.onrender.com/inference"

# Example payload (must match your API's expected fields)
payload = {
    "age": 40,
    "workclass": "Local-gov",
    "fnlgt": 215646,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

def run_inference():
    try:
        response = requests.post(API_URL, json=payload)

        print("Status code:", response.status_code)

        # If JSON is returned, print it
        try:
            print("Response JSON:", response.json())
        except Exception:
            print("No JSON returned. Raw response:")
            print(response.text)

    except Exception as e:
        print("Error sending request:", e)


if __name__ == "__main__":
    run_inference()