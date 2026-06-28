import os
import time


def dprint(*args):
    """Lightweight debug print controlled by DEBUG=1."""
    if os.environ.get("DEBUG", "1") == "1":
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}]", *args, flush=True)
