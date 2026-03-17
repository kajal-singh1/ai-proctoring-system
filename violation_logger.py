import csv
import os
import uuid
from datetime import datetime

LOG_FILE = "violations.csv"

# Session state
_session_id  = None
_session_start = None

def start_session():
    """Call this when exam begins. Returns the session ID."""
    global _session_id, _session_start
    _session_id    = str(uuid.uuid4())[:8].upper()
    _session_start = datetime.now()

    # Create log file with headers if needed
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id", "timestamp",
                "violation_type", "details"
            ])
        print(f"✅ Log file created: {LOG_FILE}")
    else:
        print(f"✅ Log file exists: {LOG_FILE}")

    print(f"🎓 Session started | ID: {_session_id} | Time: {_session_start.strftime('%H:%M:%S')}")
    return _session_id

def end_session():
    """Call this when exam ends. Prints session summary."""
    global _session_id, _session_start
    if _session_id is None:
        print("No active session.")
        return

    duration = datetime.now() - _session_start
    print(f"\n{'='*50}")
    print(f"📋 SESSION REPORT — {_session_id}")
    print(f"Duration : {str(duration).split('.')[0]}")

    # Count violations for this session
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["session_id"] == _session_id]

        counts = {}
        for row in rows:
            v = row["violation_type"]
            counts[v] = counts.get(v, 0) + 1

        if counts:
            print("Violations:")
            for v_type, count in sorted(counts.items()):
                print(f"  {v_type:<20} {count} times")
            print(f"  {'TOTAL':<20} {sum(counts.values())} times")
        else:
            print("  No violations — clean session! ✅")

    print(f"{'='*50}\n")
    _session_id = None

def log_violation(violation_type, details=""):
    """Log one violation to CSV."""
    if _session_id is None:
        print("WARNING: No active session. Call start_session() first.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([_session_id, timestamp, violation_type, details])

    print(f"[{_session_id}] {timestamp} | {violation_type} | {details}")

def init_logger():
    """Legacy support — calls start_session()."""
    start_session()