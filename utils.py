# utils.py

import json
import os
import time

# Simple logger to save each query result
def save_log_json(data, folder="logs"):
    os.makedirs(folder, exist_ok=True)
    timestamp = int(time.time())
    path = os.path.join(folder, f"log_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
