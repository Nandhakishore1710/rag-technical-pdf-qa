import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)

if response.status_code == 200:
    models = [m["id"] for m in response.json().get("data", [])]
    print("\n✅ YOUR ACCESSIBLE GROQ MODELS:")
    for m in models:
        print(f" - {m}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")