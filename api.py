import requests
import time
import random

X_ACCESS_KEY = "$2a$10$Vm5SBIfh1mmTtT8lW1kFiuY2c2vmh1QrdmcskFKBYRYYOnpIwADN6"
X_MASTER_KEY = "$2a$10$hknkGAz8HC03Co4o0ur76.CIBedxOQs3yMa1uqtrSDC/U9ISYqQL6"

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