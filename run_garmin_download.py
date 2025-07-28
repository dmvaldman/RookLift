import os
import argparse
import dotenv

from garmin_client import GarminClient


def main():
    """
    Main entry point for downloading Garmin data.
    Provides two modes via command-line arguments:
    - 'update': (Default) Incrementally updates data from the last saved date.
    - 'full':   Performs a full historical download from the first-ever data point.
    """
    parser = argparse.ArgumentParser(description="Download Garmin data.")
    parser.add_argument(
        "--mode",
        type=str,
        default="update",
        choices=["update", "full"],
        help="The download mode: 'update' (default) or 'full'."
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    email = os.environ.get("garmin_email")
    password = os.environ.get("garmin_password")

    if not email or not password:
        raise ValueError("Garmin credentials not found in environment variables.")

    client = GarminClient(email, password)

    if args.mode == "full":
        client.download_all(save=True)
    else:  # 'update' mode
        client.download(save=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")