import os
import dotenv
from supabase import create_client, Client

dotenv.load_dotenv()

# We use a global variable to cache the client. This is important to avoid
# creating a new connection for every single database call.
_db_client: Client = None


def get_db() -> Client:
    """
    Returns a Supabase client, initializing it if it doesn't exist.
    This lazy initialization is crucial for Modal, as secrets are not available
    at import time but are available when the function is called.
    """
    global _db_client
    if _db_client is not None:
        return _db_client

    # If the client doesn't exist, create it.
    # these are all available in the modal-created secret
    url: str = os.environ.get("supabase_url")
    key: str = os.environ.get("supabase_service_key")

    if not url or not key:
        raise Exception(
            "Supabase URL or service_role key not found. Please check your .env file."
        )

    _db_client = create_client(url, key)
    return _db_client