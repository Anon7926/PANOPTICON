import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_BASE_DIR = "PANOPTICON"
DEFAULT_DATASET_DIR = os.path.join(DEFAULT_BASE_DIR, "PANOPTICON_v1.4")
DEFAULT_FIG_DIR = os.path.join(DEFAULT_BASE_DIR, "Figures1.4")
DEFAULT_OUTFILE = "scenario_distribution_pie.png"
CATEGORY_OUTFILE = "category_distribution_pie.png"
NESTED_OUTFILE = "category_scenario_nested_pie.png"
PII_OUTFILE = "pii_presence_pie.png"
SCENARIO_BAR_OUTFILE = "scenario_distribution_bar.png"
PII_TABLE_OUTFILE = "pii_by_category.csv"


def load_dataset(dataset_path: str):
    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        return ds[split_name], split_name
    return ds, None


def count_scenarios(dataset) -> Counter:
    counts = Counter()
    for row in dataset:
        scenario = row.get("Scenario") if isinstance(row, dict) else None
        if scenario:
            counts[str(scenario)] += 1
    return counts


def aggregate_counts(dataset):
    category_counts = Counter()
    scenario_counts = Counter()
    scenario_by_category: Dict[str, Counter] = defaultdict(Counter)
    pii_counts = Counter({"with_pii": 0, "without_pii": 0})
    category_pii_counts: Dict[str, Counter] = defaultdict(lambda: Counter({"with_pii": 0, "without_pii": 0}))

    for row in dataset:
        category = row.get("Category") if isinstance(row, dict) else None
        scenario = row.get("Scenario") if isinstance(row, dict) else None
        pii_field = row.get("PromptPII") if isinstance(row, dict) else None

        if category:
            category_counts[str(category)] += 1
        if scenario:
            scenario_counts[str(scenario)] += 1
        if category and scenario:
            scenario_by_category[str(category)][str(scenario)] += 1

        has_pii = _has_pii(pii_field)
        if has_pii:
            pii_counts["with_pii"] += 1
        else:
            pii_counts["without_pii"] += 1

        if category:
            if has_pii:
                category_pii_counts[str(category)]["with_pii"] += 1
            else:
                category_pii_counts[str(category)]["without_pii"] += 1

    return category_counts, scenario_counts, scenario_by_category, pii_counts, category_pii_counts


def collapse_counts(counts: Counter, min_percent: float) -> Tuple[List[str], List[int], int]:
    total = sum(counts.values())
    if total == 0:
        return [], [], 0

    if min_percent <= 0:
        labels, values = zip(*counts.most_common())
        return list(labels), list(values), total

    labels: List[str] = []
    values: List[int] = []
    other_total = 0

    for name, value in counts.most_common():
        pct = (value / total) * 100
        if pct >= min_percent:
            labels.append(name)
            values.append(value)
        else:
            other_total += value

    if other_total:
        labels.append(f"Other (<{min_percent:.1f}% each)")
        values.append(other_total)

    return labels, values, total


def _autopct(total: int):
    def formatter(pct: float) -> str:
        absolute = int(round(pct * total / 100.0))
        return f"{pct:.1f}% ({absolute})"

    return formatter


def _lighten_color(color, factor: float = 0.65):
    r, g, b = mcolors.to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b


def plot_pie(labels: List[str], values: List[int], total: int, title: str, out_path: str) -> None:
    cmap = plt.cm.tab20 if len(labels) <= 20 else plt.cm.tab20b
    colors = cmap(np.linspace(0, 1, len(labels))) if labels else None

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        values,
        autopct=_autopct(total),
        startangle=90,
        colors=colors,
        textprops={"fontsize": 9},
    )
    ax.axis("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(
        wedges,
        labels,
        title="Scenario",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pie chart to: {out_path}")


def plot_category_pie(category_counts: Counter, out_path: str) -> None:
    if not category_counts:
        print("No categories found; skipping category pie chart.")
        return

    labels, values = zip(*category_counts.most_common())
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, _ = ax.pie(
        values,
        labels=labels,
        autopct=_autopct(total),
        startangle=90,
        colors=plt.cm.tab10(np.linspace(0, 1, len(labels))),
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title("Category Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved category pie chart to: {out_path}")


def plot_nested_category_scenarios(
    category_counts: Counter,
    scenario_by_category: Dict[str, Counter],
    out_path: str,
) -> None:
    if not category_counts or not scenario_by_category:
        print("Missing categories or scenarios; skipping nested pie.")
        return

    categories, cat_values = zip(*category_counts.most_common())
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    scenario_labels: List[str] = []
    scenario_values: List[int] = []
    scenario_colors: List[tuple] = []
    total = sum(cat_values)

    for idx, category in enumerate(categories):
        cat_color = base_colors[idx]
        scenarios = scenario_by_category.get(category, {})
        for scenario, value in scenarios.most_common():
            scenario_labels.append(scenario)
            scenario_values.append(value)
            scenario_colors.append(_lighten_color(cat_color))

    fig, ax = plt.subplots(figsize=(12, 12))

    # Single ring: scenarios (no labels/numbers on slices)
    ax.pie(
        scenario_values,
        radius=1.0,
        labels=None,
        colors=scenario_colors,
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )

    legend_labels = [
        f"{label}: {value} ({(value / total) * 100:.2f}%)"
        for label, value in zip(scenario_labels, scenario_values)
    ]
    handles = [Patch(facecolor=col, edgecolor="white") for col in scenario_colors]
    ax.legend(
        handles,
        legend_labels,
        title="Scenario (count, % of dataset)",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        title_fontsize=10,
    )

    ax.set(aspect="equal", title="Categories and Scenarios")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved nested category/scenario pie chart to: {out_path}")


def plot_pii_presence(pii_counts: Counter, out_path: str) -> None:
    total = sum(pii_counts.values())
    if total == 0:
        print("No rows found; skipping PII presence pie chart.")
        return

    labels = ["With PII tag", "Without PII tag"]
    values = [pii_counts["with_pii"], pii_counts["without_pii"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        autopct=_autopct(total),
        startangle=90,
        colors=["#d62728", "#1f77b4"],
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title("Prompts With vs Without PII", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PII presence pie chart to: {out_path}")


def plot_scenario_bar(scenario_counts: Counter, total: int, title: str, out_path: str) -> None:
    if not scenario_counts:
        print("No scenarios found; skipping scenario bar chart.")
        return

    items = scenario_counts.most_common()
    labels = [name for name, _ in items]
    values = [val for _, val in items]
    percents = [(v / total) * 100 for v in values]

    y_pos = np.arange(len(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.3)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for bar, val, pct in zip(bars, values, percents):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.2f}%)",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved scenario bar chart to: {out_path}")


def write_pii_table(category_pii_counts: Dict[str, Counter], out_path: str) -> None:
    if not category_pii_counts:
        print("No category/PII data; skipping table.")
        return

    lines = ["Category,With_PII,Without_PII,Total,With_PII_Percent"]
    for category, counts in category_pii_counts.items():
        with_pii = counts.get("with_pii", 0)
        without_pii = counts.get("without_pii", 0)
        total = with_pii + without_pii
        pct = (with_pii / total * 100) if total else 0.0
        lines.append(f"{category},{with_pii},{without_pii},{total},{pct:.2f}")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved PII-by-category table to: {out_path}")


def _has_pii(pii_field) -> bool:
    if pii_field is None:
        return False
    if isinstance(pii_field, dict):
        return any(val not in (None, "", []) for val in pii_field.values())
    return bool(pii_field)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a pie chart for PANOPTICON scenario distribution."
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
        help="Filename for the generated pie chart.",
    )
    parser.add_argument(
        "--min-percent",
        type=float,
        default=0.0,
        help=(
            "Minimum percentage for a scenario to appear as its own slice. "
            "Lower-frequency scenarios are grouped into 'Other' when this is > 0."
        ),
    )
    parser.add_argument(
        "--title",
        default="PANOPTICON Scenario Distribution (v1.4)",
        help="Chart title.",
    )
    args = parser.parse_args()

    dataset, split_name = load_dataset(args.dataset_path)
    category_counts, scenario_counts, scenario_by_category, pii_counts, category_pii_counts = aggregate_counts(dataset)

    if not scenario_counts:
        print("No Scenario values found in the dataset.")
        return

    labels, values, total = collapse_counts(scenario_counts, args.min_percent)

    print(f"Active split: {split_name or 'N/A'}")
    print(f"Total rows: {total}")
    print("\nTop scenarios:")
    for name, value in scenario_counts.most_common(10):
        pct = (value / total) * 100
        print(f"  {name}: {value} ({pct:.2f}%)")

    print("\nCategory counts:")
    for name, value in category_counts.most_common():
        pct = (value / total) * 100
        print(f"  {name}: {value} ({pct:.2f}%)")

    pii_total = sum(pii_counts.values())
    if pii_total:
        print(
            f"\nPII tags: with={pii_counts['with_pii']} "
            f"({(pii_counts['with_pii']/pii_total)*100:.2f}%), "
            f"without={pii_counts['without_pii']} "
            f"({(pii_counts['without_pii']/pii_total)*100:.2f}%)"
        )

    if args.min_percent > 0 and labels and labels[-1].startswith("Other ("):
        grouped = len(scenario_counts) - (len(labels) - 1)
        print(
            f"Grouped {grouped} low-frequency scenarios into '{labels[-1]}' "
            f"(threshold: {args.min_percent:.2f}%)."
        )

    os.makedirs(args.fig_dir, exist_ok=True)
    scenario_out = os.path.join(args.fig_dir, args.outfile)
    category_out = os.path.join(args.fig_dir, CATEGORY_OUTFILE)
    nested_out = os.path.join(args.fig_dir, NESTED_OUTFILE)
    pii_out = os.path.join(args.fig_dir, PII_OUTFILE)
    scenario_bar_out = os.path.join(args.fig_dir, SCENARIO_BAR_OUTFILE)
    pii_table_out = os.path.join(args.fig_dir, PII_TABLE_OUTFILE)

    plot_pie(labels, values, total, args.title, scenario_out)
    plot_scenario_bar(scenario_counts, total, "Scenario Distribution (Counts)", scenario_bar_out)
    plot_category_pie(category_counts, category_out)
    plot_nested_category_scenarios(category_counts, scenario_by_category, nested_out)
    plot_pii_presence(pii_counts, pii_out)
    write_pii_table(category_pii_counts, pii_table_out)


if __name__ == "__main__":
    main()
