import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

client = MLClient.from_config(DefaultAzureCredential())
endpoint = client.online_endpoints.get("loan-default-rf-endpoint")
keys = client.online_endpoints.get_keys("loan-default-rf-endpoint")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {keys.primary_key}"
}

data = {
    "loan_amount": 200000,
    "income": 60000,
    "property_value": 350000,
    "dtir1": 32,
    "Credit_Score": 720
}

response = requests.post(endpoint.scoring_uri, headers=headers, json=data)
print(response.json())