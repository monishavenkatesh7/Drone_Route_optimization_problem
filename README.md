# Drone_Route_optimization_problem

To run the programme:  
The input should be with the name **"input.json"** and should be in the below format

```json
{
    "city": {
        "grid_size": 20
    },
    "drones": {
        "fleet": [
            {
                "id": "D1",
                "max_payload": 20,
                "max_distance": 100,
                "speed": 2,
                "available": true
            },
            {
                "id": "D2",
                "max_payload": 25,
                "max_distance": 40,
                "speed": 1.5,
                "available": true
            },
            {
                "id": "D3",
                "max_payload": 7,
                "max_distance": 20,
                "speed": 2.5,
                "available": true
            }
        ]
    },
    "orders": [
        {
            "id": "O1",
            "delivery_x": 2,
            "delivery_y": 2,
            "deadline": 15,
            "package_weight": 2
        },
        {
            "id": "O2",
            "delivery_x": 3,
            "delivery_y": 3,
            "deadline": 30,
            "package_weight": 8
        },
        {
            "id": "O3",
            "delivery_x": 4,
            "delivery_y": 4,
            "deadline": 25,
            "package_weight": 4
        },
        {
            "id": "O4",
            "delivery_x": 13,
            "delivery_y": 12,
            "deadline": 40,
            "package_weight": 8
        }
    ]
}

Kindly make all the libraries in the requirements.txt has been installed using the below command in the terminal

'''pip install -r requirements.txt'''
and run the code using the below command

'''python code.py'''
