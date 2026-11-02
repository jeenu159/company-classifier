import json
import os
import time
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from schema import Classification, TAXONOMY
from tools import fda_profile

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini").with_structured_output(Classification)
CACHE = "data/profiles.json"

PROMPT = """Classify this pharmaceutical-industry company into exactly one class.

Classes:
{taxonomy}

Company name: {company}

FDA label profile (computed from up to 100 of the company's drug labels):
{profile}

Use the company name and the profile. Pick the single best class."""

profiles = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

with open("eval/companies_eval.jsonl", encoding="utf-8") as f:
    gold = [json.loads(line) for line in f if line.strip()]

records = []
for item in gold:
    company = item["company"]
    if company not in profiles:
        profiles[company] = fda_profile(company)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=1)

    t0 = time.perf_counter()
    try:
        out = llm.invoke(PROMPT.format(taxonomy=TAXONOMY, company=company,
                                       profile=json.dumps(profiles[company], indent=1)))
        pred, conf, evidence = out.label, out.confidence, out.evidence
    except Exception as e:
        pred, conf, evidence = "ERROR", 0.0, [str(e)[:200]]
    latency = time.perf_counter() - t0

    strict = pred == item["label"]
    lenient = strict or pred in item["also_ok"]
    records.append({"id": item["id"], "company": company, "gold": item["label"], "pred": pred,
                    "confidence": conf, "strict": strict, "lenient": lenient,
                    "evidence": evidence, "latency_s": round(latency, 2)})
    mark = "OK " if strict else ("ok~" if lenient else "XX ")
    print(f"{mark} {item['id']:>2}  {company[:40]:<40} gold={item['label']:<20} pred={pred:<20} conf={conf:.2f}")

n = len(records)
print(f"\nstrict accuracy  = {sum(r['strict'] for r in records) / n:.2f}")
print(f"lenient accuracy = {sum(r['lenient'] for r in records) / n:.2f}")

per_class = Counter(r["gold"] for r in records)
right = Counter(r["gold"] for r in records if r["strict"])
print("\nper class (strict):")
for label in sorted(per_class):
    print(f"  {label:<22} {right[label]}/{per_class[label]}")

os.makedirs("eval/runs", exist_ok=True)
run_file = f"eval/runs/baseline_{datetime.now():%Y%m%d_%H%M}.jsonl"
with open(run_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"saved to {run_file}")