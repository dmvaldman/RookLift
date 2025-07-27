import os
from supabase import create_client, Client

# Get Supabase credentials from environment variables
# For server-side operations that require bypassing RLS, use the service_role key.
url: str = os.environ.get("supabase_url")
key: str = os.environ.get("supabase_service_key")

# For client-side operations, you would use the anon key.
# anon_key: str = os.environ.get("supabase_key")

if not url or not key:
    raise Exception("Supabase URL or service_role key not found. Please check your .env file.")

# Initialize the Supabase client
db: Client = create_client(url, key)