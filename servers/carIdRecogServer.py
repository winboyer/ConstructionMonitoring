import json
import requests

# Read data from file via API with parameters
read_url = "http://112.124.54.138:5001/api/loadcars/range"
write_url = "http://112.124.54.138:5001/api/loadcars"

get_params = {
    "start": "2025-12-13T00:00:00",
    "end": "2025-12-14T00:00:00"
}
post_data = {
    "licenSeplate": "京A123478",
    "timestamp": "2025-12-13"
}

try:
    response = requests.get(read_url, params=get_params, timeout=5)
    # response = requests.post(write_url, json=post_data, timeout=5)

    # read response
    # response.raise_for_status()
    # data = response.json()
    # print(json.dumps(data, ensure_ascii=False, indent=2))
    # content = data.get("data")
    # print(f"File Content: {content}")

    #write response
    response.raise_for_status()
    print(f"Success: {response.status_code}")
    print(f"Response: {response.text}")
    
except requests.exceptions.RequestException as e:
    print(f"Read Error: {e}")
