from datasets import load_from_disk, concatenate_datasets
import os, re

# 1) Collect shard dirs: PANOPTICON_0, PANOPTICON_1, ..., PANOPTICON_6
prefix = "PANOPTICON_"
dirs = [d for d in os.listdir("..") if re.fullmatch(rf"{prefix}\d+", d)]
dirs = sorted(dirs, key=lambda d: int(d.split("_")[1]))

print("Found shards:", dirs)

# 2) Load each shard
datasets = [load_from_disk(d) for d in dirs]

# 3) Handle Dataset vs DatasetDict just in case
from datasets import Dataset, DatasetDict

if all(isinstance(ds, Dataset) for ds in datasets):
    merged = concatenate_datasets(datasets)

elif all(isinstance(ds, DatasetDict) for ds in datasets):
    # Merge each split (e.g., "train", "test") across shards
    merged = DatasetDict({
        split: concatenate_datasets([ds[split] for ds in datasets])
        for split in datasets[0].keys()
    })

else:
    raise TypeError("Mixed Dataset and DatasetDict shards; normalize them first.")

# 4) Save unified dataset
out_dir = "PANOPTICON_FULL"
merged.save_to_disk(out_dir)
print(f"Saved merged dataset to: {out_dir}")
