import os
import requests
from dotenv import load_dotenv

# .env dosyasındaki GEMINI_API_KEY'i al (boşlukları temizle)
load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("HATA: .env dosyasında GEMINI_API_KEY bulunamadı!")
    exit()

api_key = api_key.strip()
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print("Google API'den güncel modeller çekiliyor...\n")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    print("--- LITELLM İÇİN GEÇERLİ METİN (CHAT) MODELLERİ ---")
    for model in data.get('models', []):
        if 'generateContent' in model.get('supportedGenerationMethods', []):
            # 'models/' kısmını LiteLLM'in istediği 'gemini/' formatına çeviriyoruz
            print(f"gemini/{model['name'].split('/')[-1]}")
            
    print("\n--- LITELLM İÇİN GEÇERLİ EMBEDDING MODELLERİ ---")
    for model in data.get('models', []):
        if 'embedContent' in model.get('supportedGenerationMethods', []):
            print(f"gemini/{model['name'].split('/')[-1]}")
else:
    print(f"API Hatası: {response.status_code} - {response.text}")