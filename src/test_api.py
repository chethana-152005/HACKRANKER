import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Fetching available models...")
try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"Available Model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTesting generation with 'gemini-1.5-flash'...")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello")
    print(f"Success: {response.text}")
except Exception as e:
    print(f"Failed: {e}")