import csv
import os
from datetime import datetime

LOG_FILE = "violations.csv"

def init_logger():
    """Create the CSV file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "violation_type", "details"])
        print(f"✅ Log file created: {LOG_FILE}")
    else:
        print(f"✅ Log file already exists: {LOG_FILE}")

def log_violation(violation_type, details=""):
    """Append one violation row to the CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, violation_type, details])
    print(f"[VIOLATION LOGGED] {timestamp} | {violation_type} | {details}")