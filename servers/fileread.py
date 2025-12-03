import json
import requests

# Read data from file via API with parameters
read_url = "http://120.26.34.95:9001/api/easy/read"
params = {
    "fileName": "塔吊实时数据.json",
    "encoding": "utf-8"
}

try:
    response = requests.get(read_url, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    content = data.get("content")
    print(f"File Content: {content}")
    
except requests.exceptions.RequestException as e:
    print(f"Read Error: {e}")