import os
from dotenv import load_dotenv
import unify

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv('UNIFY_API_KEY')

if not api_key:
    raise ValueError("API key not found. Ensure 'UNIFY_API_KEY' is set in the environment or .env file.")

def get_current_credits(api_key):
    """
    Fetch and return the current remaining credits.

    Args:
        api_key (str): The API key for accessing Unify API.

    Returns:
        str: The remaining credits or an error message.
    """
    try:
        credits = unify.get_credits(api_key=api_key)
        return f"Credits remaining: ${credits}"
    except Exception as e:
        return f"Error fetching credits: {e}"

def list_available_endpoints(api_key):
    """
    Fetch and return the list of available endpoints.

    Args:
        api_key (str): The API key for accessing Unify API.

    Returns:
        str: A formatted string of available endpoints or an error message.
    """
    try:
        endpoints = unify.list_endpoints(api_key=api_key)
        if endpoints:
            return "Available endpoints:\n" + "\n".join(f"- {endpoint}" for endpoint in endpoints)
        else:
            return "No endpoints available."
    except Exception as e:
        return f"Error fetching endpoints: {e}"

def main():
    """
    Demonstrate the use of get_current_credits and list_available_endpoints.
    """
    print("Fetching credits...")
    print(get_current_credits(api_key))
    print("\nFetching available endpoints...")
    print(list_available_endpoints(api_key))

if __name__ == "__main__":
    main()
