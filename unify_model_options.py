import unify
from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()

api_key = os.getenv('API_KEY')

print(unify.list_endpoints(api_key=api_key))
