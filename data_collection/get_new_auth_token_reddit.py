import base64
import json
import os
import requests

id = "QkUsWUsdKxz5ZW0Cw_JmlQ"
secret = "dgPACtoDgLVOlnz7FkAxxFxAvHJVIg"
basicAuth = base64.b64encode(f"{id}:{secret}".encode("utf-8")).decode("utf-8")

data = {
    "grant_type": "password",
    "username": "NeighborhoodHungry60",
    "password": os.getenv("REDDIT_PASSWORD")
}

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; PPC Mac OS X 10_8_7 rv:5.0; en-US) AppleWebKit/533.31.5 (KHTML, like Gecko) Version/4.0 Safari/533.31.5',
    "Authorization": f"Basic {basicAuth}"
}

response = requests.post("https://www.reddit.com/api/v1/access_token", headers=headers, data=data)

print(response.text)