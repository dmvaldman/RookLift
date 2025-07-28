import os
import datetime
import time

from garminconnect import Garmin
from db import db

# A small delay to be respectful to the Garmin API
API_DELAY = 0.2


class GarminClient:
    def __init__(self, email, password):
        self.client = Garmin(email, password)
        self.client.login()

    def download_stress(self, start_date, end_date):
        """Downloads daily stress data for the given date range."""
        data = []
        for day in range((end_date - start_date).days + 1):
            date = start_date + datetime.timedelta(days=day)
            stress_data = self.client.get_stress_data(date.isoformat())
            stress_avg = stress_data.get('avgStressLevel')

            data.append({
                "date": stress_data.get('calendarDate'),
                "stress_avg": int(stress_avg) if stress_avg is not None and stress_avg != -1 else None
            })
            time.sleep(API_DELAY)
        return data

    def download_body_battery_sleep(self, start_date, end_date):
        """Downloads body battery change during sleep for the given date range."""
        data = []
        for day in range((end_date - start_date).days + 1):
            date = start_date + datetime.timedelta(days=day)
            summary = self.client.get_user_summary(date.isoformat())

            bb_during_sleep = summary.get('bodyBatteryDuringSleep')

            data.append({
                "date": summary.get('calendarDate'),
                "body_battery_during_sleep": int(bb_during_sleep) if bb_during_sleep is not None else None
            })
            time.sleep(API_DELAY)
        return data

    def download_daily_summary(self, start_date, end_date):
        """Downloads a summary of daily metrics for the given date range."""
        data = []
        start_date_adj = start_date - datetime.timedelta(days=1)
        end_date_adj = end_date - datetime.timedelta(days=1)

        for day in range((end_date_adj - start_date_adj).days + 1):
            date = start_date_adj + datetime.timedelta(days=day)
            summary = self.client.get_user_summary(date.isoformat())
            date_shifted = datetime.datetime.strptime(summary['calendarDate'], '%Y-%m-%d').date() + datetime.timedelta(days=1)

            steps = summary.get('totalSteps')
            sedentary_duration = summary.get('sedentarySeconds')
            stress_duration = summary.get('highStressDuration')
            low_stress_duration = summary.get('lowStressDuration')
            active_calories = summary.get('activeKilocalories')

            data.append({
                "date": str(date_shifted),
                "steps": int(steps) if steps is not None else None,
                "sedentary_duration": int(sedentary_duration) if sedentary_duration is not None else None,
                "stress_duration": int(stress_duration) if stress_duration is not None else None,
                "low_stress_duration": int(low_stress_duration) if low_stress_duration is not None else None,
                "active_calories": int(active_calories) if active_calories is not None else None,
            })
            time.sleep(API_DELAY)
        return data

    def download_body_battery(self, start_date, end_date):
        """Downloads the maximum daily body battery for the given date range."""
        def get_chunk(start, end):
            values = []
            bb_data = self.client.get_body_battery(start.isoformat(), end.isoformat())
            for item in bb_data:
                battery_values = [v[1] for v in item['bodyBatteryValuesArray'] if v[1] is not None]
                if battery_values:
                    values.append({"date": item['date'], "battery_max": max(battery_values)})
            return values

        all_data = []
        current_start = start_date
        while current_start <= end_date:
            current_end = min(current_start + datetime.timedelta(days=29), end_date)
            all_data.extend(get_chunk(current_start, current_end))
            current_start += datetime.timedelta(days=30)
            time.sleep(API_DELAY)
        return all_data

    def download_sleep(self, start_date, end_date):
        """Downloads detailed sleep metrics for the given date range."""
        data = []
        for day in range((end_date - start_date).days + 1):
            date = start_date + datetime.timedelta(days=day)
            sleep_data = self.client.get_sleep_data(date.isoformat())
            dto = sleep_data.get('dailySleepDTO', {})
            if 'sleepScores' in dto:
                data.append({
                    "date": dto['calendarDate'],
                    "sleep_stress": dto.get('avgSleepStress'),
                    "light_duration": dto.get('lightSleepSeconds'),
                    "rem_duration": dto.get('remSleepSeconds'),
                    "deep_duration": dto.get('deepSleepSeconds'),
                    "sleep_duration": dto.get('sleepTimeSeconds'),
                    "sleep_score": dto.get('sleepScores', {}).get('overall', {}).get('value'),
                    "awake_duration": dto.get('awakeSleepSeconds')
                })
            time.sleep(API_DELAY)
        return data

    def download_activities(self, start_date, end_date):
        """Downloads total calories burned during activities for the given date range."""
        def get_chunk(start, end):
            activities_by_date = {}
            activities = self.client.get_activities_by_date(start.isoformat(), end.isoformat())
            for activity in activities:
                date_str = (datetime.datetime.strptime(activity['startTimeLocal'].split(' ')[0], '%Y-%m-%d').date() + datetime.timedelta(days=1)).isoformat()
                activities_by_date.setdefault(date_str, 0)

                calories = activity.get('calories')
                if calories is not None:
                    activities_by_date[date_str] += calories

            # Round the summed calories and cast to an integer to match the DB schema.
            return [{"date": d, "activity_calories": int(round(c))} for d, c in activities_by_date.items() if c is not None]

        all_data = []
        current_start = start_date - datetime.timedelta(days=1)
        current_end = end_date - datetime.timedelta(days=1)

        while current_start <= current_end:
            chunk_end = min(current_start + datetime.timedelta(days=29), current_end)
            all_data.extend(get_chunk(current_start, chunk_end))
            current_start += datetime.timedelta(days=30)
            time.sleep(API_DELAY)
        return all_data

    def save_to_db(self, data, table_name):
        """Saves a list of data dictionaries to the specified Supabase table."""
        if not data:
            print(f"No data to save for table '{table_name}'.")
            return None
        try:
            print(f"Saving {len(data)} records to '{table_name}'...")
            result = db.table(table_name).upsert(data, on_conflict="date").execute()
            return result
        except Exception as e:
            print(f"❌ Error saving to '{table_name}': {e}")
            return None

    def download_range(self, start_date, end_date, save=False):
        """Downloads all Garmin signals for the date range and optionally saves them."""
        print(f"🚀 Starting download of all Garmin data from {start_date} to {end_date}...")

        # A mapping of download functions to the target table name
        download_map = {
            self.download_stress: "garmin_stress",
            self.download_sleep: "garmin_sleep",
            self.download_activities: "garmin_activities",
            self.download_body_battery: "garmin_body_battery",
            self.download_body_battery_sleep: "garmin_body_battery_sleep",
            self.download_daily_summary: "garmin_summary",
        }

        all_data = {}
        for download_func, table_name in download_map.items():
            print(f"\n--- Downloading {table_name} data ---")
            data = download_func(start_date, end_date)
            if save:
                self.save_to_db(data, table_name)
            all_data[table_name] = data

        if save:
            print("\n🎉 All Garmin data downloaded and saved successfully.")

        return all_data

    def download_all(self, save=True):
        """
        Performs a full historical download of all Garmin data.
        1. Finds the user's first-ever date with Garmin data.
        2. Downloads all signals from that date up to the present.
        """
        print("🚀 Starting full historical download of all Garmin data...")
        start_date = self.get_first_date()
        if not start_date:
            print("❌ Could not find any Garmin data for this user. Exiting.")
            return None

        end_date = datetime.date.today()

        print(f"\nFound first data on {start_date}. Proceeding to download all history up to {end_date}.")

        return self.download_range(start_date, end_date, save=save)

    def download(self, save=True):
        """
        Incrementally updates all Garmin data in the database.
        Checks for the last saved date and downloads all new signals.
        If the database is empty, it performs a full historical download.
        """
        print("🚀 Starting smart update of all Garmin data...")
        last_date = self.get_last_recorded_date()

        if not last_date:
            print("\n- No existing Garmin data found in the database. Starting full download.")
            return self.download_all(save=save)

        start_date = last_date + datetime.timedelta(days=1)
        end_date = datetime.date.today()

        print(f"🔍 Last saved date is {last_date}. Resuming download from {start_date} to {end_date}.")

        if start_date > end_date:
            print("✅ Your database is already up-to-date. No new data to download.")
            return None

        return self.download_range(start_date, end_date, save=save)

    def get_first_date(self, start_date=None, end_date=None):
        """
        Finds the earliest date with available Garmin data using a binary search.

        Args:
            start_date (datetime.date, optional): The earliest date to check. Defaults to 2 years ago.
            end_date (datetime.date, optional): The latest date to check. Defaults to today.

        Returns:
            A datetime.date object for the first day with data, or None.
        """
        print("🔍 Searching for the first day of Garmin data...")
        if end_date is None:
            end_date = datetime.date.today()
        if start_date is None:
            start_date = end_date - datetime.timedelta(days=365 * 2)

        # Base case: if the range is a single day, check it.
        if (end_date - start_date).days <= 1:
            # Check the end date first as it's more likely to have data
            if self.client.get_body_battery(end_date.isoformat()):
                return end_date
            elif self.client.get_body_battery(start_date.isoformat()):
                return start_date
            return None

        mid_date = start_date + (end_date - start_date) // 2

        # Check if there's any body battery data at the midpoint.
        # This is a good proxy for general data availability.
        data = self.client.get_body_battery(mid_date.isoformat())

        if data and data[0].get('bodyBatteryValuesArray'):
            # Data exists, so the first day might be earlier.
            return self.get_first_date(start_date, mid_date)
        else:
            # No data, so the first day must be later.
            return self.get_first_date(mid_date, end_date)

    def get_last_recorded_date(self):
        """
        Finds the most recent date across all Garmin tables in the database.

        Returns:
            A datetime.date object for the last recorded date, or None if all tables are empty.
        """
        print("🔍 Finding the last recorded date for Garmin data in the database...")
        latest_date = None
        garmin_tables = [
            "garmin_stress", "garmin_sleep", "garmin_activities",
            "garmin_body_battery", "garmin_body_battery_sleep", "garmin_summary"
        ]

        for table in garmin_tables:
            try:
                result = db.table(table).select("date").order("date", desc=True).limit(1).execute()
                if result.data:
                    current_date = datetime.datetime.strptime(result.data[0]['date'], "%Y-%m-%d").date()
                    if latest_date is None or current_date > latest_date:
                        latest_date = current_date
            except Exception as e:
                print(f"⚠️ Could not query table '{table}': {e}")
                continue

        if latest_date:
            print(f"✅ Last recorded date found: {latest_date}")
        else:
            print("No Garmin data found in the database.")

        return latest_date