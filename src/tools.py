import json
import re
import sys
import time
from collections import Counter
import requests
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langsmith import traceable

load_dotenv()

FDA_URL = "https://api.fda.gov/drug/label.json"
SPF = re.compile(r"\bSPF\s?\d+")                            # sunscreen claims: SPF 30, SPF50
MADE_BY = re.compile(r"(?:MANUFACTURED|MFD\.?|MADE)\s+(?:BY|FOR)[:\s]+([A-Z0-9][A-Z0-9 .,&'()-]{2,50})")
SKIP_WORDS = {"THE", "INC", "LLC", "CO", "COMPANY", "CORP"}


def _all_text(label):
    """Every text field on the label (not the openfda metadata), as one uppercase string."""
    parts = []
    for key, value in label.items():
        if key != "openfda" and isinstance(value, list):
            parts.extend(v for v in value if isinstance(v, str))
    return " ".join(parts).upper()


def _key_token(company):
    """First distinctive word of the company name, used to spot 'made by someone else'."""
    for word in re.findall(r"[A-Z0-9][A-Z0-9-]+", company.upper()):
        if word not in SKIP_WORDS:
            return word
    return company.upper()


def _fetch_labels(company, sample):
    """Fetch labels with retries. openFDA sometimes returns 500 on large responses, so shrink the page."""
    last_err = None
    for limit in (sample, 50, 25):
        for attempt in range(3):
            try:
                resp = requests.get(
                    FDA_URL,
                    params={"search": f'openfda.manufacturer_name.exact:"{company}"', "limit": limit},
                    timeout=60,
                )
                if resp.status_code == 404:
                    return None                       # no labels for this company
                resp.raise_for_status()               # 429 / 5xx raise here and get retried
                return resp.json()
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(2 ** attempt)              # 1s, 2s, 4s
    raise last_err

@traceable(run_type="tool")
def fda_profile(company: str, sample: int = 100) -> dict:
    """Summarise a company's FDA drug labels into evidence for classification."""
    data = _fetch_labels(company, sample)
    if data is None:
        return {"company": company, "found": False}
    labels = data["results"]
    key = _key_token(company)

    product_types, routes, brands, other_makers = Counter(), Counter(), Counter(), Counter()
    flags = Counter()
    rx_total = rx_brand_is_generic = 0

    for label in labels:
        fda = label.get("openfda", {})
        ptype = (fda.get("product_type") or ["UNKNOWN"])[0]
        product_types[ptype] += 1
        for r in fda.get("route", []):
            routes[r] += 1
        brand = (fda.get("brand_name") or [""])[0]
        generic = (fda.get("generic_name") or [""])[0]
        if brand:
            brands[brand] += 1
        if ptype == "HUMAN PRESCRIPTION DRUG":
            rx_total += 1
            if brand and generic and brand.strip().upper() == generic.strip().upper():
                rx_brand_is_generic += 1

        text = _all_text(label)
        if "HOMEOPATHIC" in text or "HPUS" in text:
            flags["homeopathic"] += 1
        if SPF.search(text):
            flags["sunscreen"] += 1
        if "HAND SANITIZER" in text:
            flags["hand_sanitizer"] += 1
        if any(w in text for w in ("REPACKAGED", "RELABELED", "REPACKAGER", "RELABELER")):
            flags["repackaged_text"] += 1

        makers = [m.strip(" .,") for m in MADE_BY.findall(text)]
        others = [m for m in makers if key not in m]
        if makers and others and not any(key in m for m in makers):
            flags["made_by_others"] += 1
            other_makers[others[0][:40]] += 1

    n = len(labels)
    return {
        "company": company,
        "found": True,
        "total_labels_at_fda": data["meta"]["results"]["total"],
        "labels_sampled": n,
        "product_types": dict(product_types),
        "rx_share_brand_equals_generic": round(rx_brand_is_generic / rx_total, 2) if rx_total else None,
        "share_made_by_other_companies": round(flags["made_by_others"] / n, 2),
        "other_makers_named": [m for m, _ in other_makers.most_common(6)],
        "share_homeopathic": round(flags["homeopathic"] / n, 2),
        "share_sunscreen": round(flags["sunscreen"] / n, 2),
        "share_hand_sanitizer": round(flags["hand_sanitizer"] / n, 2),
        "share_repackaged_text": round(flags["repackaged_text"] / n, 2),
        "top_routes": [r for r, _ in routes.most_common(3)],
        "sample_brand_names": [b for b, _ in brands.most_common(12)],
    }


_tavily = None

@traceable(run_type="tool")
def web_search(company: str, max_results: int = 3) -> dict:
    """Search the web for what a company does. Returns short snippets, not full pages."""
    global _tavily
    if _tavily is None:
        _tavily = TavilySearch(max_results=max_results)
    query = f'"{company}" company business what does it do'
    r = _tavily.invoke(query)
    return {
        "company": company,
        "query": query,
        "results": [
            {"title": x["title"], "url": x["url"], "snippet": x["content"][:400]}
            for x in r.get("results", [])
        ],
    }


if __name__ == "__main__":
    if sys.argv[1] == "--web":
        print(json.dumps(web_search(" ".join(sys.argv[2:])), indent=1))
    else:
        print(json.dumps(fda_profile(" ".join(sys.argv[1:])), indent=1))