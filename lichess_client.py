import lichess.api
import datetime
import requests
from collections import defaultdict

from db import db
from utils import (
    datetime_to_timestamp,
    timestamp_to_datetime,
    adjust_date_for_day_start,
)


class Lichess:
    def __init__(self, username):
        self.username = username
        self.timezone = "US/Pacific"
        self.start_hour = 6  # 6am local time

    def get_first_date(self, game_type="Blitz"):
        """
        Fetches rating history from the Lichess API to find the earliest date
        a rated game was played for a specific game type.

        Args:
            game_type (str): The game type to search for (e.g., "Blitz", "Rapid").

        Returns:
            A datetime.date object representing the earliest game date, or None.
        """
        print(f"🔍 Searching Lichess history for '{self.username}' to find the first '{game_type}' game date...")
        url = f"https://lichess.org/api/user/{self.username}/rating-history"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
        except requests.RequestException as e:
            print(f"❌ Error fetching rating history from Lichess: {e}")
            return None

        earliest_date_for_type = None

        for game_type_history in data:
            # Find the history for the specific game type we're interested in.
            if game_type_history.get("name") == game_type:
                points = game_type_history.get("points", [])
                if not points:
                    break  # Found the type, but no games played.

                # The first entry in the points list is the earliest game.
                year, month, day, _ = points[0]
                # Lichess API month is 0-indexed, so add 1.
                earliest_date_for_type = datetime.date(year, month + 1, day)
                break  # Exit loop once we've found our game type.

        if earliest_date_for_type:
            print(f"✅ Found the earliest '{game_type}' game date: {earliest_date_for_type}")
        else:
            print(f"❌ No rated '{game_type}' game history found for this user.")

        return earliest_date_for_type

    def download_ratings(self, start_date, end_date, game_type='Blitz'):
        """Download rating data from Lichess API and return as list of dictionaries."""
        perfType = game_type.lower()

        since = datetime_to_timestamp(start_date, timezone=self.timezone)
        until = datetime_to_timestamp(end_date, timezone=self.timezone)
        game_generator = lichess.api.user_games(
            self.username, perfType=perfType, rated=True, since=since, until=until
        )

        ratings = []
        for game in game_generator:
            timestamp_ms = game['createdAt']
            date = timestamp_to_datetime(timestamp_ms, timezone=self.timezone)

            if game['players']['white']['user']['name'] == self.username:
                rating_before = game['players']['white']['rating']
                rating_after = rating_before + game['players']['white']['ratingDiff']
            else:
                rating_before = game['players']['black']['rating']
                rating_after = rating_before + game['players']['black']['ratingDiff']

            data_formatted = {
                "date": date,
                "rating_before": rating_before,
                "rating_after": rating_after,
            }
            ratings.append(data_formatted)

        ratings.sort(key=lambda x: x['date'])

        daily_ratings_dict = defaultdict(
            lambda: {"rating_before": None, "rating_after": None, "date": None}
        )
        for record in ratings:
            adjusted_date = adjust_date_for_day_start(
                record["date"], start_hour=self.start_hour
            ).date()

            record_time = record["date"]

            if (
                daily_ratings_dict[adjusted_date].get("first_game_time") is None
                or record_time < daily_ratings_dict[adjusted_date]["first_game_time"]
            ):
                daily_ratings_dict[adjusted_date]["rating_before"] = record["rating_before"]
                daily_ratings_dict[adjusted_date]["first_game_time"] = record_time

            if (
                daily_ratings_dict[adjusted_date].get("last_game_time") is None
                or record_time >= daily_ratings_dict[adjusted_date]["last_game_time"]
            ):
                daily_ratings_dict[adjusted_date]["rating_after"] = record["rating_after"]
                daily_ratings_dict[adjusted_date]["last_game_time"] = record_time

        daily_ratings = [
            {
                "date": str(date),
                "rating_morning": data["rating_before"],
                "rating_evening": data["rating_after"],
            }
            for date, data in daily_ratings_dict.items()
        ]

        return daily_ratings

    def save_ratings_to_db(self, ratings_data):
        """Save ratings data to Supabase using upsert."""
        if not ratings_data:
            print("No data to save.")
            return None

        try:
            # By specifying `on_conflict="date"`, we tell Supabase to update
            # the existing row if a row with the same date already exists.
            result = db.table("lichess").upsert(ratings_data, on_conflict="date").execute()
            return result
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            print("\n🔍 Troubleshooting:")
            print("1. Ensure the `lichess` table exists.")
            print("2. Run the `supabase/initial_schema.sql` script in your Supabase SQL Editor.")
            print("3. Check your .env file for correct Supabase credentials.")
            return None

    def download_and_save_ratings(self, start_date, end_date, game_type='Blitz'):
        """Download ratings data and save it to Supabase."""
        print(
            f"Downloading Lichess ratings for '{self.username}' from {start_date} to {end_date}..."
        )

        ratings_data = self.download_ratings(start_date, end_date, game_type)

        if ratings_data:
            print(f"Downloaded data for {len(ratings_data)} days. Saving to database...")
            result = self.save_ratings_to_db(ratings_data)
            if result:
                print("✅ Successfully saved ratings data to database.")
            return result
        else:
            print("No new rating data found for the specified date range.")
            return None

    def get_ratings_from_db(self, start_date=None, end_date=None):
        """Retrieve ratings data from Supabase for a given date range."""
        query = db.table("lichess").select("*")

        if start_date:
            query = query.gte("date", str(start_date))
        if end_date:
            query = query.lte("date", str(end_date))

        result = query.order("date", desc=False).execute()
        return result.data

    def get_last_recorded_date(self):
        """
        Retrieves the most recent date from the 'lichess' table.

        Returns:
            A datetime.date object representing the last recorded date,
            or None if the table is empty.
        """
        try:
            result = (
                db.table("lichess")
                .select("date")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                last_date_str = result.data[0]["date"]
                return datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
            else:
                return None  # No records found
        except Exception as e:
            print(f"❌ Error fetching last recorded date: {e}")
            return None