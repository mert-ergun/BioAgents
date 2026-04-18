import json

import requests

genes = [
    "APOE",
    "BIN1",
    "TREM2",
    "CLU",
    "PICALM",
    "ABCA7",
    "CD33",
    "MS4A6A",
    "CR1",
    "INPP5D",
    "GFAP",
    "NEFL",
    "VGF",
    "CHIT1",
    "YKL40",
    "TREM2",
    "APP",
    "MAPT",
    "SOD1",
    "PRNP",
]
db = "KEGG_2021_Human"

# Submit gene list to Enrichr
url = "https://maayanlab.cloud/Enrichr/addList"
payload = {"list": (None, "\n".join(genes)), "description": (None, "BioAgents query")}
resp = requests.post(url, files=payload, timeout=30)
resp.raise_for_status()
data = resp.json()
user_list_id = data["userListId"]

# Get enrichment results
url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={db}"
resp = requests.get(url, timeout=30)
resp.raise_for_status()
enrich_data = resp.json()

results = []
for _lib_name, terms in enrich_data.items():
    for term in terms[:20]:
        results.append(
            {
                "term": term[1],
                "pvalue": term[2],
                "adjusted_pvalue": term[6],
                "odds_ratio": term[3],
                "combined_score": term[4],
                "overlapping_genes": term[5],
            }
        )

results.sort(key=lambda x: x["pvalue"])
print(json.dumps(results[:15], indent=2))
