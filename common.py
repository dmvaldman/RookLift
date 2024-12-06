from modal import Volume, Image, Secret, is_local, Cron, App

app = App("RookLift")

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
secrets = Secret.from_dotenv()
vol = Volume.from_name("RookLift-Data", create_if_missing=True)
is_local = is_local()