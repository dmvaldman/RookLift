#!/usr/bin/env python3

import os
import datetime
import dotenv
from lichess_client import Lichess

# Load environment variables
dotenv.load_dotenv()


def download_lichess_data(game_type="Blitz"):
    """
    Incrementally downloads Lichess ratings to the database.

    Checks for the last saved date and downloads all new data from that point
    up to the current day. If the database is empty, it performs a full
    download from the user's first-ever game.
    """
    print(f"🚀 Starting incremental download for '{game_type}' ratings...")

    # Get configuration from environment
    lichess_username = os.environ.get("lichess_username")
    if not lichess_username:
        raise ValueError("lichess_username not found in environment variables")

    # Initialize Lichess client
    client = Lichess(lichess_username)

    # 1. Check for the last saved date in the database.
    last_date = client.get_last_recorded_date()

    if last_date:
        # If data exists, start downloading from the next day.
        start_date = last_date + datetime.timedelta(days=1)
        print(f"🔍 Last saved date is {last_date}. Resuming download from {start_date}.")
    else:
        # If the database is empty, find the first-ever game to start from.
        print("📋 No data in database. Finding first-ever game to begin full download.")
        start_date = client.get_first_date(game_type=game_type)
        if not start_date:
            print(f"❌ Could not find any game history for '{game_type}'. Exiting.")
            return

    # 2. The end date is always today.
    end_date = datetime.date.today()

    # 3. Check if there's a valid period to download.
    if start_date > end_date:
        print("✅ Database is already up-to-date. No new data to download.")
        return

    print(f"Downloading ratings for {lichess_username} from {start_date} to {end_date}")

    # 4. Download and save the new data.
    result = client.download_and_save_ratings(
        start_date=start_date, end_date=end_date, game_type=game_type
    )

    if result:
        print("✅ Data successfully downloaded and saved to Supabase!")

        # Optionally, retrieve and display some stats
        data = client.get_ratings_from_db(start_date, end_date)
        if data:
            ratings = [
                r["rating_evening"] for r in data if r["rating_evening"] is not None
            ]
            if ratings:
                print(f"📊 Downloaded {len(data)} days of data")
                print(f"📊 Rating range: {min(ratings)} - {max(ratings)}")
                print(f"📊 Latest rating: {ratings[-1] if ratings else 'N/A'}")
    else:
        print("❌ No new data was downloaded.")


if __name__ == "__main__":
    try:
        # You can change the default game type here, e.g., "Rapid"
        download_lichess_data(game_type="Blitz")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("  lichess_username=your_lichess_username")
        print("  supabase_url=your_supabase_project_url")
        print("  supabase_service_key=your_supabase_service_key")