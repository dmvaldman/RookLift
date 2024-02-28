from modal import Stub, Volume, Image, Secret, is_local, Cron

stub = Stub("RookLift")

image = Image.debian_slim().pip_install_from_requirements("requirements.txt")
secrets = Secret.from_dotenv()
vol = Volume.persisted("RookLift")
is_local = is_local()