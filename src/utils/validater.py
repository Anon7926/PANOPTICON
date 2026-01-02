#!/usr/bin/env python

"""
PANOPTICON dataset validation script (no Response column).

Usage:
    python validate_panopticon.py

Assumes the merged HuggingFace dataset is saved in:
    PANOPTICON/
in the current working directory.

Expected columns:
    - ID
    - Category
    - Scenario
    - Prompt
    - PromptPII
    - Content
"""

from datasets import load_from_disk, Dataset, DatasetDict
from collections import Counter
import pandas as pd
import random
import math
import sys
from pprint import pprint

DATASET_DIR = "../PANOPTICON"


def safe_len(x):
    """Return len(x) if possible, else 0."""
    try:
        return len(x)
    except TypeError:
        return 0


def main():
    print("=" * 80)
    print(f"Loading dataset from: {DATASET_DIR}")
    print("=" * 80)

    ds = load_from_disk(DATASET_DIR)

    # Handle Dataset vs DatasetDict (just in case)
    if isinstance(ds, DatasetDict):
        print("Detected DatasetDict. Available splits:", list(ds.keys()))
        if "train" in ds:
            d = ds["train"]
            active_split = "train"
        else:
            first_split = list(ds.keys())[0]
            d = ds[first_split]
            active_split = first_split
        print(f"Using split for validation: {active_split}")
    elif isinstance(ds, Dataset):
        d = ds
        print("Detected single Dataset object.")
    else:
        print("Unknown dataset type:", type(ds))
        sys.exit(1)

    print("\n--- BASIC INFO ---")
    print(d)
    print(f"Number of rows: {len(d)}")
    print(f"Features: {d.column_names}")

    # Convert to pandas for easier checks
    print("\nConverting to pandas DataFrame...")
    df = d.to_pandas()
    print("DataFrame shape:", df.shape)

    print("\n--- FIRST FEW ROWS ---")
    print(df.head(3))

    # Null / missing values
    print("\n--- NULL / MISSING VALUES PER COLUMN ---")
    print(df.isnull().sum())

    # Check columns exist (soft assert)
    expected_cols = ["ID", "Category", "Scenario", "Prompt", "PromptPII", "Content"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("\nWARNING: Missing expected columns:", missing)

    # Scenario distribution
    if "Scenario" in df.columns:
        print("\n--- SCENARIO COUNTS ---")
        scenario_counts = Counter(df["Scenario"])
        pprint(scenario_counts)
    else:
        print("\nNo 'Scenario' column found; skipping scenario counts.")

    # Category distribution
    if "Category" in df.columns:
        print("\n--- CATEGORY COUNTS ---")
        category_counts = Counter(df["Category"])
        pprint(category_counts)
    else:
        print("\nNo 'Category' column found; skipping category counts.")

    # ID checks
    if "ID" in df.columns:
        print("\n--- ID CHECKS ---")
        print("Unique IDs:", df["ID"].nunique())
        dup_ids = df.duplicated(subset=["ID"]).sum()
        print("Duplicate IDs:", dup_ids)
    else:
        print("\nNo 'ID' column; skipping ID checks.")

    # Text length stats
    if "Prompt" in df.columns:
        print("\n--- PROMPT LENGTH STATS (characters) ---")
        df["prompt_length"] = df["Prompt"].astype(str).str.len()
        print(df["prompt_length"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]))
    else:
        print("\nNo 'Prompt' column; skipping prompt length stats.")

    if "Content" in df.columns:
        print("\n--- CONTENT LENGTH STATS (characters) ---")
        df["content_length"] = df["Content"].astype(str).str.len()
        print(df["content_length"].describe(percentiles=[0.1, 0.5, 0.9, 0.99]))
    else:
        print("\nNo 'Content' column; skipping content length stats.")

    print("\n--- DUPLICATE CHECKS ---")
    if "Prompt" in df.columns:
        dup_prompts = df.duplicated(subset=["Prompt"]).sum()
        print(f"Duplicate Prompts: {dup_prompts}")
    else:
        print("No 'Prompt' column; skipping duplicate-prompt check.")

    if "Prompt" in df.columns and "Content" in df.columns:
        # Make sure everything is hashable by converting to strings
        tmp = df.copy()
        tmp["__Prompt_str"] = tmp["Prompt"].astype(str)
        tmp["__Content_str"] = tmp["Content"].astype(str)
        dup_pairs = tmp.duplicated(subset=["__Prompt_str", "__Content_str"]).sum()
        print(f"Duplicate (Prompt, Content) pairs (string-wise): {dup_pairs}")
    else:
        print("Missing 'Prompt' or 'Content'; skipping duplicate pair check.")

    # PromptPII coverage
    pii_col = "PromptPII" if "PromptPII" in df.columns else None
    print("\n--- PII COVERAGE ---")
    if pii_col:
        empty_pii = df[pii_col].apply(lambda x: safe_len(x) == 0 or pd.isna(x)).sum()
        non_empty_pii = len(df) - empty_pii
        print(f"Rows with EMPTY/NO {pii_col}: {empty_pii}")
        print(f"Rows with NON-EMPTY {pii_col}: {non_empty_pii}")
        if non_empty_pii > 0:
            # Distribution of number of PII tokens
            num_tokens = df[pii_col].apply(lambda x: safe_len(x))
            print("\nNumber of PII items per row:")
            print(num_tokens.describe(percentiles=[0.1, 0.5, 0.9, 0.99]))
    else:
        print("No 'PromptPII' column; skipping PII coverage checks.")

    # Random sample inspection
    print("\n--- RANDOM SAMPLE INSPECTION ---")
    n_samples = min(5, len(d))
    indices = random.sample(range(len(d)), n_samples) if len(d) >= n_samples else list(range(len(d)))

    for idx in indices:
        row = d[idx]
        print("\n" + "-" * 60)
        print(f"Index: {idx}")
        if "ID" in row:
            print(f"ID: {row['ID']}")
        if "Scenario" in row:
            print(f"Scenario: {row['Scenario']}")
        if "Category" in row:
            print(f"Category: {row['Category']}")
        if "Prompt" in row:
            print("\nPROMPT:")
            print(row["Prompt"])
        if "Content" in row:
            print("\nCONTENT:")
            print(row["Content"])
        if pii_col and pii_col in row:
            print(f"\n{pii_col}: {row[pii_col]}")

    print("\n" + "=" * 80)
    print("Validation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()

