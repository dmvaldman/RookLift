import json
from common import stub, vol

@stub.function(volumes={"/data": vol})
def main():
    # load file
    with open('/data/daily_battery.json', 'r') as f:
        print(json.load(f))