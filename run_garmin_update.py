import os
import datetime
import dotenv

from garmin_client import GarminClient


def update_garmin_data():
    """
    Incrementally updates all Garmin data in the database.

    Checks for the last saved date and downloads all new signals from that point
    up to the current day. If the database is empty, it advises the user to
    run the full download script first.
    """
    print("🚀 Starting incremental update of all Garmin data...")

    dotenv.load_dotenv()
    email = os.environ.get("garmin_email")
    password = os.environ.get("garmin_password")

    if not email or not password:
        raise ValueError("Garmin credentials not found in environment variables.")

    client = GarminClient(email, password)

    # 1. Check for the last saved date in the database.
    last_date = client.get_last_recorded_date()

    if not last_date:
        print("\n❌ No existing Garmin data found in the database.")
        print("💡 Please run `run_garmin_full_download.py` first to populate your history.")
        return

    # 2. Start downloading from the day after the last record.
    start_date = last_date + datetime.timedelta(days=1)
    end_date = datetime.date.today()

    print(f"🔍 Last saved date is {last_date}. Resuming download from {start_date} to {end_date}.")

    # 3. Check if there is actually a new period to download.
    if start_date > end_date:
        print("✅ Your database is already up-to-date. No new data to download.")
        return

    # 4. Perform the incremental download and save all signals.
    client.download_and_save_all(start_date, end_date)

    print("\n🎉 Garmin data update finished.")


if __name__ == "__main__":
    try:
        update_garmin_data()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")