import os
import datetime
import dotenv

from lichess_client import Lichess

# Load environment variables from .env file
dotenv.load_dotenv()

def download_historical_ratings(game_type="Blitz"):
    """
    Performs a historical download of Lichess ratings, stopping 90 days ago.

    This script is designed for testing the incremental update functionality.
    It populates the database with older data, leaving a recent gap that
    `run_local_update.py` can then fill in.
    """
    print(f"🚀 Starting historical download of '{game_type}' ratings (ending 90 days ago)...")

    lichess_username = os.environ.get("lichess_username")
    client = Lichess(lichess_username)

    start_date = client.get_first_date(game_type=game_type)
    end_date = datetime.date.today() - datetime.timedelta(days=90)

    # 3. Perform the historical download and save the data.
    client.download_and_save_ratings(start_date, end_date, game_type=game_type)


if __name__ == "__main__":
    try:
        download_historical_ratings(game_type="Blitz")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")