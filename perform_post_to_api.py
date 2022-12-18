import json

import requests

HEROKU_ADDRESS = "https://mlops-api-udacity.herokuapp.com/inference"

SAMPLE_INPUT = {
    "age": 42,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}


def main():
    data = json.dumps(SAMPLE_INPUT)

    response = requests.post(HEROKU_ADDRESS, data)
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")


if __name__ == '__main__':
    main()
