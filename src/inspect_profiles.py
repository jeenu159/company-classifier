import json
import sys

profiles = json.load(open("data/profiles.json", encoding="utf-8"))
names = sys.argv[1:] or [
    "Bryant Ranch Prepack", "A-S Medication Solutions", "NuCare Pharmaceuticals,Inc.",
    "PD-Rx Pharmaceuticals, Inc.", "Direct_Rx",
]
for name in names:
    p = profiles.get(name, {})
    print(f"{name[:30]:<30} made_by_others={p.get('share_made_by_other_companies')}  "
          f"repackaged={p.get('share_repackaged_text')}  "
          f"brand=generic={p.get('rx_share_brand_equals_generic')}  "
          f"types={p.get('product_types')}")