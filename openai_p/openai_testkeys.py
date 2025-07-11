import os
import requests
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Use separate keys for basic and admin access
BASIC_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-your-basic-key"
ADMIN_API_KEY = os.getenv("OPENAI_ADMIN_KEY") or "sk-your-admin-key"
PROJECT_ID = os.getenv("OPENAI_DEFAULTPROJECT_ID") or "proj-your-default-project-id"

BASE_URL = "https://api.openai.com"

def test_basic_access():
    print("🔍 Testing basic access (/v1/models)...")
    headers = {
        "Authorization": f"Bearer {BASIC_API_KEY}"
    }
    res = requests.get(f"{BASE_URL}/v1/models", headers=headers)
    print("Status:", res.status_code)
    print("Response:", res.json())

def test_admin_access():
    print(f"\n🔐 Testing admin access (https://api.openai.com/v1/organization/projects/{PROJECT_ID}/api_keys)...")
    headers = {
        "Authorization": f"Bearer {ADMIN_API_KEY}"
    }
    res = requests.get(f"https://api.openai.com/v1/organization/projects/{PROJECT_ID}/api_keys", headers=headers)
    print(headers)  # Print the API key used for admin access
    print("Status:", res.status_code)
    print("Response:", res.json())

if __name__ == "__main__":
    test_basic_access()
    test_admin_access()
