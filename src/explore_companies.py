import json
import os
import requests

URL = "https://api.fda.gov/drug/label.json"
OUT = "data/raw/manufacturers.json"

resp = requests.get(URL, params={"count": "openfda.manufacturer_name.exact", "limit": 1000}, timeout=60)
resp.raise_for_status()
rows = resp.json()["results"]   # [{"term": name, "count": n}, ...]

os.makedirs("data/raw", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=1)

print(f"{len(rows)} manufacturers (top 1000 by label count)\n")
for r in rows[:40]:
    print(f"{r['count']:>6}  {r['term']}")