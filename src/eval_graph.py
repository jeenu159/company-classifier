import json
import os
import time
import uuid
from datetime import datetime
from graph import build_graph

graph = build_graph()

with open("eval/companies_eval.jsonl", encoding="utf-8") as f:
    gold = [json.loads(line) for line in f if line.strip()]

records = []
for item in gold:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.perf_counter()
    try:
        r = graph.invoke({"company": item["company"]}, config)
        pred, status = r["classification"]["label"], r["status"]
        used_web, attempts, trace = r["used_web"], r["attempts"], r["trace"]
        first = trace[1].split(": ")[1].split(" ")[0] if len(trace) > 1 else pred   # label from classify #1
    except Exception as e:
        pred, status, used_web, attempts, first = "ERROR", "error", False, 0, "ERROR"
        trace = [f"{type(e).__name__}: {e}"[:300]]
    latency = time.perf_counter() - t0

    ok = lambda p: p == item["label"] or p in item["also_ok"]
    rec = {"id": item["id"], "company": item["company"], "gold": item["label"], "pred": pred,
           "first_pred": first, "strict": pred == item["label"], "lenient": ok(pred),
           "first_lenient": ok(first), "status": status, "used_web": used_web,
           "attempts": attempts, "latency_s": round(latency, 2), "trace": trace}
    records.append(rec)
    mark = "OK " if rec["strict"] else ("ok~" if rec["lenient"] else "XX ")
    print(f"{mark} {item['id']:>2}  {item['company'][:36]:<36} gold={item['label']:<20} pred={pred:<20} "
          f"{status:<12} web={'y' if used_web else 'n'} tries={attempts}")

n = len(records)
auto = [r for r in records if r["status"] == "auto"]
review = [r for r in records if r["status"] == "needs_review"]
retried = [r for r in records if r["attempts"] > 1]

print(f"\nstrict accuracy            = {sum(r['strict'] for r in records) / n:.2f}")
print(f"lenient accuracy           = {sum(r['lenient'] for r in records) / n:.2f}")
print(f"with humans fixing queue   = {(sum(r['lenient'] for r in auto) + len(review)) / n:.2f}   (lenient)")
print(f"auto-accepted              = {len(auto)}/{n}, accuracy {sum(r['lenient'] for r in auto) / max(len(auto), 1):.2f}")
print(f"sent to human review       = {len(review)}/{n}")
print(f"silent errors (auto, wrong)= {sum(not r['lenient'] for r in auto)}   {[r['id'] for r in auto if not r['lenient']]}")
print(f"used web search            = {sum(r['used_web'] for r in records)}/{n}")
print(f"validator forced a retry   = {len(retried)}/{n}")
print(f"retry fixed the answer     = {sum(r['lenient'] and not r['first_lenient'] for r in retried)}")
print(f"retry broke the answer     = {sum(r['first_lenient'] and not r['lenient'] for r in retried)}")
print(f"median latency             = {sorted(r['latency_s'] for r in records)[n // 2]:.1f}s")

os.makedirs("eval/runs", exist_ok=True)
run_file = f"eval/runs/graph_{datetime.now():%Y%m%d_%H%M}.jsonl"
with open(run_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"saved to {run_file}")