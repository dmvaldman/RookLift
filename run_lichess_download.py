import os
import argparse
import dotenv

from lichess_client import Lichess


def main():
    """
    Main entry point for downloading Lichess ratings data.
    Provides two modes via command-line arguments:
    - 'update': (Default) Incrementally updates data from the last saved date.
    - 'full':   Performs a full historical download from the first-ever game.
    """
    parser = argparse.ArgumentParser(description="Download Lichess ratings data.")
    parser.add_argument(
        "--game-type",
        type=str,
        default="Blitz",
        help="The game type to download, e.g., 'Blitz', 'Rapid', 'Bullet'."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="update",
        choices=["update", "full"],
        help="The download mode: 'update' (default) or 'full'."
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    lichess_username = os.environ.get("lichess_username")
    if not lichess_username:
        raise ValueError("Lichess username not found in environment variables.")

    client = Lichess(lichess_username)

    if args.mode == "full":
        client.download_all(game_type=args.game_type, save=True)
    else:  # 'update' mode
        client.download(game_type=args.game_type, save=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")