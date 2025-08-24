import json
import os
import logging

from modal_defs import image, secrets, vol, app, Cron, is_local
from create_dataset import create_dataset
from create_model import save_model, good_baseline, analyze, preprocess

import garth
garth.http.USER_AGENT = {"User-Agent": ("GCM-iOS-5.7.2.1")}

logging.basicConfig(level=logging.INFO)

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vol},
    schedule=Cron("0 12 * * 1"),
    timeout=600
)
def download_and_create():
    save = True
    model_type = 'LogisticRegression'
    # model_type = 'LogisticRegressionSparse'

    features = [
        'active_calories',
        # 'activity_calories',
        # 'awake_duration',
        # 'battery_max',
        # 'body_battery',
        'deep_duration',
        'stress_duration',
        'light_duration',
        # 'low_stress_duration',
        'rem_duration',
        # 'sedentary_duration',
        # 'sleep_duration',
        # 'sleep_score',
        # 'sleep_stress',
        # 'steps',
        'stress_avg'
    ]

    # Determine if running locally or on Modal
    save_dir = "data" if is_local else "/data"
    save_path_df = os.path.join(save_dir, "fitness_signals.csv")
    save_path_df_processed = os.path.join(save_dir, "fitness_signals_processed.csv")
    save_path_model = os.path.join(save_dir, "model_data.json")
    save_path_baseline = os.path.join(save_dir, "model_ranges.json")

    print("\n--- Creating Unified Dataset ---")
    df = create_dataset()

    if save:
        df.to_csv(save_path_df)
        print(f"💾 Unified dataset saved to {save_path_df}")

    df = preprocess(df, features=features, include_rating_cols=True, num_days_lag=0, aggregate_activity=False, save=save, save_path=save_path_df_processed)

    model, scaler, column_names = analyze(df, model_type=model_type, plot=False)
    ranges = good_baseline(df)

    if save:
        save_model(model, scaler, column_names, save_path=save_path_model)
        with open(save_path_baseline, 'w') as f:
            json.dump(ranges, f, indent=2)

        if not is_local:
            vol.commit()


if __name__ == '__main__':
    download_and_create.local()
