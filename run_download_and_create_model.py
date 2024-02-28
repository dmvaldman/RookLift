import json
import os
from garminconnect import Garmin
from common import stub, image, secrets, vol, is_local, Cron

from download import download
from create_model import save_model, good_baseline, analyze, preprocess

# Runs every Monday at 5am PT
@stub.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vol},
    schedule=Cron("0 12 * * 1")
)
def download_and_create():
    save = True
    plot = False

    save_dir = "data" if is_local else "/data"
    save_path_df = save_dir + "/fitness_signals.csv"
    save_path_df_processed = save_dir + "/fitness_signals_processed.csv"
    save_path_model = save_dir + "/model_data.json"
    save_path_baseline = save_dir + "/model_ranges.json"

    lichess_username = os.getenv("lichess_username")
    garmin_email = os.getenv('garmin_email')
    garmin_password = os.getenv('garmin_password')

    garmin = Garmin(garmin_email, garmin_password)
    garmin.login()

    df = download(lichess_username, garmin, save=save, save_dir=save_dir, save_path=save_path_df)
    df = preprocess(df, save=save, save_path=save_path_df_processed)

    model, scaler, column_names = analyze(df, plot=plot)
    ranges = good_baseline(df)

    if save:
        save_model(model, scaler, column_names, save_path=save_path_model)
        with open(save_path_baseline, 'w') as f:
            json.dump(ranges, f)

    vol.commit()

if __name__ == '__main__':
    download_and_create.local()
