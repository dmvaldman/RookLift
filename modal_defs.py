import os
import modal

Cron = modal.Cron

is_local = not bool(os.environ.get("MODAL_POD_NAME"))

# Define the base container image.
# We use .add_local_python_source() to explicitly include all our .py files
# in the container. This makes sure imports work correctly when deployed.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .workdir("/root")
    .add_local_dir(".", remote_path="/root")
)

# Define the secret to load from the .env file
secrets = modal.Secret.from_dotenv(".env")

# Define the shared data volume
vol = modal.Volume.from_name("RookLift-Data", create_if_missing=True)

app = modal.App("RookLift")