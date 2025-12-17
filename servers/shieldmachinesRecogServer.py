import json
import requests

# Read data from file via API with parameters
read_url = "http://112.124.54.138:5000/api/shieldmachines_1/latest"
write_url = "http://112.124.54.138:5000/api/shieldmachines_1"

post_data = {
    "code": 0,
    "data": {
        "id": 2,
        "shieldMachineId": "D108",
        "ringNumber": 1001,
        "timestamp": "2025-12-08T10:00:00",
        "status": "Running",
        "cutterHeadMonitoring": {
            "rotationSpeed": 2.5,
            "torque": 1500,
            "power": 750,
            "overBreakAmount": 0.2,
            "cutterHeadAngle": 45.5,
            "cutterWear": 15.3
        },
        "propulsionSystem": {
            "pitchAngle": -1.2,
            "rollAngle": 0.8,
            "penetrationRate": 85.5,
            "propulsionPressure": 300,
            "propulsionSpeed": 45.2,
            "totalThrust": 25000,
            "gripperHeadAngle": 90,
            "gripperHeadPressure": 150
        },
        "cylinderGroups": [
            {
                "name": "A组油缸",
                "rightTopEarthPressure": null,
                "rightCenterEarthPressure": 1.5,
                "rightBottomEarthPressure": null,
                "leftBottomEarthPressure": null,
                "leftCenterEarthPressure": null,
                "leftTopEarthPressure": null,
                "groupPropulsionDisplacement": 1200,
                "groupPropulsionPressure": 150
            },
            {
                "name": "B组油缸",
                "rightTopEarthPressure": null,
                "rightCenterEarthPressure": null,
                "rightBottomEarthPressure": 1.8,
                "leftBottomEarthPressure": 1.7,
                "leftCenterEarthPressure": null,
                "leftTopEarthPressure": null,
                "groupPropulsionDisplacement": 1180,
                "groupPropulsionPressure": 145.5
            },
            {
                "name": "C组油缸",
                "rightTopEarthPressure": null,
                "rightCenterEarthPressure": null,
                "rightBottomEarthPressure": null,
                "leftBottomEarthPressure": null,
                "leftCenterEarthPressure": 1,
                "leftTopEarthPressure": null,
                "groupPropulsionDisplacement": 1190,
                "groupPropulsionPressure": 148.2
            },
            {
                "name": "D组油缸",
                "rightTopEarthPressure": null,
                "rightCenterEarthPressure": null,
                "rightBottomEarthPressure": null,
                "leftBottomEarthPressure": null,
                "leftCenterEarthPressure": null,
                "leftTopEarthPressure": 1.4,
                "groupPropulsionDisplacement": 1210,
                "groupPropulsionPressure": 152.1
            }
        ]
    },
    "table": "nomal_shield_machine_1"
}

try:
    # response = requests.get(read_url, timeout=5)
    response = requests.post(write_url, json=post_data, timeout=5)

    # read response
    # response.raise_for_status()
    # data = response.json()
    # print(json.dumps(data, ensure_ascii=False, indent=2))
    # content = data.get("data")
    # print(f"File Content: {content}")

    #write response
    response.raise_for_status()
    print(f"Success: {response.status_code}")
    print(f"Response: {response.content}")
    print(f"Response: {response.message}")
    
    
except requests.exceptions.RequestException as e:
    print(f"Read Error: {e}")
