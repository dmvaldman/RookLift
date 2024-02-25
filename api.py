import requests
import time
import random
import os
import dotenv

dotenv.load_dotenv()

X_ACCESS_KEY = os.getenv('JSONBIN_ACCESS_KEY')
X_MASTER_KEY = os.getenv('JSONBIN_MASTER_KEY')

url = "https://api.jsonbin.io/v3/b/65cc1fd01f5677401f2ef548"

def read():
    headers = {
        "X-Access-Key": X_ACCESS_KEY
    }

    response = requests.get(url, headers=headers)
    response_data = response.json()

    print(response_data['record'])

def write():
    headers = {
        "X-Master-Key": X_MASTER_KEY,
        "X-Access-Key": X_ACCESS_KEY,
        "Content-Type": "application/json"
    }

    # level is random int
    level = random.randint(1, 6)

    data = {
        "level": level,
        "advice": "You should take a break"
    }

    response = requests.put(url, headers=headers, json=data)
    response_data = response.json()

    print(response_data)

write()
time.sleep(1)
read()