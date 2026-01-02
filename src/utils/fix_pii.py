from datasets import load_from_disk
from PanoHandler import PanoPlus
import os
from pii_handler import match_pii


BASE_DIR = "PANOPTICON"
DATASET_PATH = os.path.join(BASE_DIR, "PANOPTICON_v1.3")
OUT_PATH = os.path.join(BASE_DIR, "PANOPTICON_v1.4")

dataset = load_from_disk(DATASET_PATH)
panop = PanoPlus()
ids = panop.get_ids()

uid_to_pii = {row["Unique ID"]: row for row in panop.dataset}

def fix_example(ex):
    uid = ex["ID"]
    ex["PromptPII"] = match_pii(ex["Prompt"], uid_to_pii[uid])
    return ex

ds_updated = dataset.map(fix_example)

# Save to disk (new folder; keeps original intact)
ds_updated.save_to_disk(OUT_PATH)

print(f"Saved updated dataset to: {OUT_PATH}")
