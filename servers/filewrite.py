import json
import requests
from datetime import datetime

# JSON data to write
data = {
    "设备ID": "TC001",
    "高度": 68.5,
    "吊重": 15.8,
    "时间": "2025-11-27 14:32:10"
}
# data = "设备ID_高度_吊重_时间"

# Send POST request
url = "http://120.26.34.95:9001/api/easy/write"
headers = {"Content-Type": "application/json"}
params = {
    "FileName": "塔吊实时数据.json",
    "IsBase64": False
}
try:
    # Prepare the request body with JSON data
    params["Content"] = json.dumps(data)
    response = requests.post(url, json=params, headers=headers)
    
    response.raise_for_status()
    print(f"Success: {response.status_code}")
    print(f"Response: {response.text}")
    
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

