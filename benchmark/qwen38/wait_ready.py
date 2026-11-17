import sys
import time

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:31038/health"
for _ in range(300):
    try:
        if requests.get(url, timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Server did not become healthy; inspect server log")
