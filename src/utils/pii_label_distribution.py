import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_BASE_DIR = "PANOPTICON"
DEFAULT_DATASET_DIR = os.path.join(DEFAULT_BASE_DIR, "PANOPTICON_v1.4")
DEFAULT_FIG_DIR = os.path.join(DEFAULT_BASE_DIR, "Figures1.4")
DEFAULT_OUTFILE = "pii_label_distribution_angled.png"
DEFAULT_LABEL_ANGLE = 35

# Ordered list matches the PII keys used across the dataset
PII_LABELS = (
    "Address",
    "Age",
    "Allergies",
    "Annual Salary",
    "Birth City",
    "Birth Date",
    "Blood Type",
    "Children Count",
    "Credit Score",
    "Disability",
    "Driver's License",
    "Education Info",
    "Email Address",
    "Emergency Contact Name",
    "Emergency Contact Phone",
    "Employer",
    "Facebook",
    "Father's Name",
    "Finance Status",
    "First Name",
    "Gender",
    "Instagram",
    "Job Title",
    "Last Name",
    "LinkedIn",
    "Locale",
    "Marital Status",
    "Mother's Name",
    "National ID",
    "Nationality",
    "Net Worth",
    "Passport Number",
    "Phone Number",
    "Pinterest",
    "Reddit",
    "Snapchat",
    "Spouse Name",
    "TikTok",
    "Twitter",
    "Work Email",
    "Work Phone",
    "Unique ID",
)


def load_dataset(dataset_path: str) -> Tuple[Dataset, str]:
    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        return ds[split_name], split_name
    return ds, None


def count_pii_labels(dataset) -> List[Tuple[str, int]]:
    counts = {label: 0 for label in PII_LABELS}
    total = len(dataset)

    for idx, row in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"Counting PII labels: {(idx / total) * 100:.2f}%", end="\r")

        pii_field = row.get("PromptPII") if isinstance(row, dict) else None
        if not pii_field:
            continue

        for label in pii_field.keys():
            if pii_field[label] is not None:
                counts[label] = counts.get(label, 0) + 1

    print("\nPII label counts:")
    print(counts)

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [(label, count) for label, count in sorted_items if count > 0]


def plot_pii_distribution(data: List[Tuple[str, int]], angle: float, out_path: str) -> None:
    if not data:
        print("No PII labels with counts; skipping plot.")
        return

    labels, values = zip(*data)

    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=angle, ha="right", rotation_mode="anchor")
    plt.xlabel("PII Labels")
    plt.ylabel("Counts")
    plt.title("PII Label Distribution")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.95)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PII label distribution to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the PII label distribution bar chart with angled x-axis labels."
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_DIR,
        help="Path to the PANOPTICON dataset folder (load_from_disk format).",
    )
    parser.add_argument(
        "--fig-dir",
        default=DEFAULT_FIG_DIR,
        help="Directory to store the generated figure.",
    )
    parser.add_argument(
        "--outfile",
        default=DEFAULT_OUTFILE,
        help="Filename for the generated chart.",
    )
    parser.add_argument(
        "--label-angle",
        type=float,
        default=DEFAULT_LABEL_ANGLE,
        help="Rotation angle (degrees) for the x-axis labels.",
    )
    args = parser.parse_args()

    dataset, split_name = load_dataset(args.dataset_path)
    print(f"Loaded split: {split_name or 'N/A'} | rows: {len(dataset)}")

    label_counts = count_pii_labels(dataset)
    out_path = os.path.join(args.fig_dir, args.outfile)
    plot_pii_distribution(label_counts, args.label_angle, out_path)


if __name__ == "__main__":
    main()
