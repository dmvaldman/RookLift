import os
import datetime
import pandas as pd
import dotenv

from db import get_db

dotenv.load_dotenv()

GARMIN_TABLES = [
    "garmin_stress",
    "garmin_body_battery",
    "garmin_body_battery_sleep",
    "garmin_sleep",
    "garmin_activities",
    "garmin_summary",
]

# This list now defines the columns to check for the final filtering step.
# It should match the Garmin-related columns returned by the get_daily_signals function.
GARMIN_COLUMN_NAMES = [
    "stress_avg", "battery_max", "body_battery", "sleep_stress",
    "light_duration", "rem_duration", "deep_duration", "sleep_duration",
    "sleep_score", "awake_duration", "activity_calories", "steps",
    "sedentary_duration", "stress_duration", "low_stress_duration", "active_calories"
]

def get_earliest_date(tables: list[str]) -> datetime.date | None:
    """Finds the earliest date available across a list of tables."""
    db = get_db()
    earliest_date = None
    for table in tables:
        try:
            # We only need to check one garmin table as per the assumption
            result = db.table(table).select("date").order("date", desc=False).limit(1).execute()
            if result.data:
                current_date = datetime.datetime.strptime(result.data[0]['date'], "%Y-%m-%d").date()
                if earliest_date is None or current_date < earliest_date:
                    earliest_date = current_date
        except Exception as e:
            print(f"⚠️ Could not query table '{table}' for earliest date: {e}")
            continue
    return earliest_date


def create_dataset(start_date=None, end_date=None):
    """
    Creates a unified dataset by fetching pre-joined data from the
    `get_daily_signals` database function.
    """
    db = get_db()
    print("📞 Calling the `get_daily_signals` database function...")

    # Set parameters for the RPC call
    params = {}

    if start_date is None:
      start_date = get_earliest_date([GARMIN_TABLES[0]])

    if end_date is None:
      end_date = datetime.date.today()

    start_date = start_date.isoformat()
    end_date = end_date.isoformat()

    params["start_date"] = start_date
    params["end_date"] = end_date

    try:
        response = db.rpc('get_daily_signals', params).execute()

        if not response.data:
            print("❌ No data returned from the database function.")
            return pd.DataFrame()

        # 3. Load data into a DataFrame
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])

        # 4. Apply final filter: Drop rows where all Garmin data is missing
        print("  - Dropping rows with no Garmin data...")

        # Dynamically determine the Garmin columns by excluding the known Lichess columns.
        # This is more robust than maintaining a hardcoded list.
        lichess_cols = ['date', 'rating_morning', 'rating_evening']
        garmin_cols = [col for col in df.columns if col not in lichess_cols]

        df.dropna(subset=garmin_cols, how='all', inplace=True)

        if df.empty:
            print("❌ No rows remaining after filtering for Garmin data.")
            return pd.DataFrame()

        # 5. Finalize the DataFrame
        df.set_index('date', inplace=True)
        print("\n✅ Successfully created the dataset.")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df

    except Exception as e:
        print(f"❌ An error occurred while calling the database function: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    try:
        final_df = create_dataset()
        if not final_df.empty:
            print("\n📊 First 5 rows of the final dataset:")
            print(final_df.head())

            save_path = "data/fitness_signals.csv"
            final_df.to_csv(save_path)
            print(f"\n💾 Dataset saved to {save_path}")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")