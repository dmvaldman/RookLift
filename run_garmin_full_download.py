import os
import datetime
import dotenv

from garmin_client import GarminClient


def download_all_garmin_data():
    """
    Performs a full historical download of all Garmin data.

    1. Finds the user's first-ever date with Garmin data.
    2. Downloads all signals from that date up to the present.

    This script is recommended for populating the database for the first time.
    """
    print("🚀 Starting full historical download of all Garmin data...")

    dotenv.load_dotenv()
    email = os.environ.get("garmin_email")
    password = os.environ.get("garmin_password")

    if not email or not password:
        raise ValueError("Garmin credentials not found in environment variables.")

    client = GarminClient(email, password)

    # 1. Find the first date with available data.
    start_date = client.get_first_date()
    if not start_date:
        print("❌ Could not find any Garmin data for this user. Exiting.")
        return

    # 2. The end date is set to 100 days ago to test incremental updates.
    end_date = datetime.date.today() - datetime.timedelta(days=100)

    print(f"\nFound first data on {start_date}. Proceeding to download history until {end_date}.")
    print("This will leave the last 30 days empty, so you can test the update script.")

    # 3. Perform the full download and save all signals.
    client.download_and_save_all(start_date, end_date)

    print("\n🎉 Full Garmin data download finished.")
    print("💡 You can now use `run_garmin_update.py` for daily incremental updates.")


if __name__ == "__main__":
    try:
        download_all_garmin_data()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")