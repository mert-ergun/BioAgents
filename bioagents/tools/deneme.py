"""Utility script to list available Gemini models via Google API."""

import os

import requests  # noqa: S113
from dotenv import load_dotenv


load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("HATA: .env dosyasinda GEMINI_API_KEY bulunamadi!")  # noqa: RUF001
    raise SystemExit(1)

api_key = api_key.strip()
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print("Google API'den guncel modeller cekiliyor...\n")

response = requests.get(url, timeout=30)  # noqa: S113

if response.status_code == 200:
    data = response.json()

    print("--- LITELLM ICIN GECERLI METIN (CHAT) MODELLERI ---")
    for model in data.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            
            print(f"gemini/{model['name'].split('/')[-1]}")

    print("\n--- LITELLM ICIN GECERLI EMBEDDING MODELLERI ---")
    for model in data.get("models", []):
        if "embedContent" in model.get("supportedGenerationMethods", []):
            print(f"gemini/{model['name'].split('/')[-1]}")
else:
    print(f"API Hatasi: {response.status_code} - {response.text}")  # noqa: RUF001
