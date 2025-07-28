# This file is the entrypoint for Modal deployments.
# It's purpose is to import the main app object and all the functions
# that are part of that app, so they can be discovered by `modal deploy`.

# Import the app object first.
from modal_defs import app

# Then, import the modules that define our functions.
# This is crucial so that the @app.function decorators in those files
# are executed and the functions get registered with the `app`.
import run_download_and_create_model
import run_predict