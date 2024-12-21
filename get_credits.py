import os
import unify
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')
# print(api_key)
credits = unify.get_credits(api_key=api_key)
print(f"Credits remaining: ${credits}")