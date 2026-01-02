import os
import re
from collections import Counter
from random import sample
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

BASE_DIR = "PANOPTICON"
DATASET_PATH = os.path.join(BASE_DIR, "PANOPTICON_v1.4")
FIG_DIR = os.path.join(BASE_DIR, "Figures1.4")

dataset = load_from_disk(DATASET_PATH)


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _ensure_nltk_stopwords():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")



def dataset_diversity(dataset):
    total = len(dataset)
    unique_set = set()
    uniques = []
    duplicates = []
    duplicate_dict = {}
    categories_dict = {}

    for x, row in enumerate(dataset):
        if row["Prompt"] not in unique_set:
            unique_set.add(row["Prompt"])
            uniques.append(row["Prompt"])
        else:
            duplicates.append(row["Prompt"])
            duplicate_dict[row["Prompt"]] = duplicate_dict.get(row["Prompt"], 0) + 1
            categories_dict[row["Category"]] = categories_dict.get(row["Category"], 0) + 1

        if x % 1000 == 0:
            print(f"Duplicate handle: {(x / total) * 100:.2f}%", end="\r")

    print("\n=== Duplicate Summary ===")
    print(f"Total Prompts: {total}")
    print(f"Unique Prompts: {len(uniques)}")
    print("---------------------")
    print(f"Duplicate Prompts: {len(duplicates)}")
    print("---------------------")
    print(f"Single Duplicate Texts (unique duplicate strings): {len(set(duplicates))}")
    print("---------------------")
    if duplicate_dict:
        most_dup = max(duplicate_dict, key=duplicate_dict.get)
        print(f"Most Duplicated Prompt: {most_dup} ({duplicate_dict[most_dup]})")
    else:
        print("Most Duplicated Prompt: None (no duplicates)")
    print("---------------------")
    print(f"Categories with Duplicates: {categories_dict.items()}")
    print("---------------------")
    print("Full duplicate dict (prompt -> count):")
    print(duplicate_dict)
    print("---------------------")

    return uniques, duplicates, duplicate_dict, categories_dict


def handle_pii_labels(dataset):
    labels = {
        "Address": 0,
        "Age": 0,
        "Allergies": 0,
        "Annual Salary": 0,
        "Birth City": 0,
        "Birth Date": 0,
        "Blood Type": 0,
        "Children Count": 0,
        "Credit Score": 0,
        "Disability": 0,
        "Driver's License": 0,
        "Education Info": 0,
        "Email Address": 0,
        "Emergency Contact Name": 0,
        "Emergency Contact Phone": 0,
        "Employer": 0,
        "Facebook": 0,
        "Father's Name": 0,
        "Finance Status": 0,
        "First Name": 0,
        "Gender": 0,
        "Instagram": 0,
        "Job Title": 0,
        "Last Name": 0,
        "LinkedIn": 0,
        "Locale": 0,
        "Marital Status": 0,
        "Mother's Name": 0,
        "National ID": 0,
        "Nationality": 0,
        "Net Worth": 0,
        "Passport Number": 0,
        "Phone Number": 0,
        "Pinterest": 0,
        "Reddit": 0,
        "Snapchat": 0,
        "Spouse Name": 0,
        "TikTok": 0,
        "Twitter": 0,
        "Work Email": 0,
        "Work Phone": 0,
        "Unique ID": 0,
    }

    total = len(dataset)
    for x, row in enumerate(dataset):
        if x % 1000 == 0:
            print(f"PII Label Handle: {(x / total) * 100:.2f}%", end="\r")

        for label in row["PromptPII"].keys():
            if row["PromptPII"][label] is not None:
                labels[label] += 1

    print("\nPII Label Counts:")
    print(labels)

    # Sort labels by count
    sorted_items = sorted(labels.items(), key=lambda x: x[1], reverse=True)
    # Drop labels that never appear in any prompt
    filtered_items = [(k, v) for k, v in sorted_items if v > 0]
    labels_sorted = [k for k, v in filtered_items]
    values_sorted = [v for k, v in filtered_items]

    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels_sorted)))
    plt.bar(labels_sorted, values_sorted, color=colors)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.xlabel("PII Labels")
    plt.ylabel("Counts")
    plt.title("PII Label Distribution")

    ensure_fig_dir()
    out_path = os.path.join(FIG_DIR, "pii_label_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[PII Labels] Figure saved to: {out_path}")
    return labels_sorted



def lexical_diversity_metrics(dataset, max_prompts=20000):
    print("[Lexical] Computing lexical diversity metrics...")
    total = len(dataset)
    limit = min(total, max_prompts)

    texts = [dataset[i]["Prompt"] for i in range(limit)]

    tokens = []
    for t in texts:
        t = t.lower()
        # Keep alphanumerics and a few special chars, everything else -> space
        t = re.sub(r"[^a-z0-9@._]+", " ", t)
        tokens.extend(t.split())

    num_tokens = len(tokens)
    vocab = set(tokens)
    num_types = len(vocab)

    if num_tokens == 0:
        print("[Lexical] No tokens found.")
        return {
            "num_tokens": 0,
            "num_types": 0,
            "ttr": 0.0,
            "token_entropy_bits": 0.0,
        }

    ttr = num_types / num_tokens

    counts = Counter(tokens)
    counts_arr = np.array(list(counts.values()), dtype=float)
    probs = counts_arr / counts_arr.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    print(f"[Lexical] Tokens: {num_tokens}")
    print(f"[Lexical] Types: {num_types}")
    print(f"[Lexical] Type-Token Ratio (TTR): {ttr:.4f}")
    print(f"[Lexical] Token entropy: {entropy:.4f} bits")

    return {
        "num_tokens": num_tokens,
        "num_types": num_types,
        "ttr": float(ttr),
        "token_entropy_bits": float(entropy),
    }


def compute_sbert_embeddings(
    dataset, sample_size=5000, model_name="sentence-transformers/all-mpnet-base-v2"
):
    total = len(dataset)
    indices = list(range(total))

    if sample_size is not None and sample_size < total:
        indices = sample(indices, sample_size)

    texts = [dataset[i]["Prompt"] for i in indices]
    categories = [dataset[i]["Category"] for i in indices]

    print(f"[SBERT] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[SBERT] Encoding {len(texts)} prompts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine distance simplifies to 1 - dot
    )

    return np.array(indices), texts, categories, embeddings


def sbert_diversity_metrics(
    dataset, sample_size=5000, model_name="sentence-transformers/all-mpnet-base-v2"
):
    print("[SBERT] Computing SBERT-based diversity metrics...")
    indices, texts, categories, emb = compute_sbert_embeddings(
        dataset, sample_size=sample_size, model_name=model_name
    )

    n = emb.shape[0]
    if n < 2:
        print("[SBERT] Not enough samples for diversity metrics.")
        return {
            "avg_pairwise_cosine_distance": None,
            "redundancy_rate_cosine<0.05": None,
            "num_samples": n,
        }

    # Approximate average pairwise distance by sampling pairs
    max_pairs = min(50000, n * (n - 1) // 2)
    idx1 = np.random.randint(0, n, size=max_pairs)
    idx2 = np.random.randint(0, n, size=max_pairs)
    cos_sim = np.sum(emb[idx1] * emb[idx2], axis=1)  # normalized embeddings
    cos_dist = 1.0 - cos_sim
    avg_pairwise_dist = float(np.mean(cos_dist))

    print(
        f"[SBERT] Approx. average pairwise cosine distance (0=identical, 2=opposite): "
        f"{avg_pairwise_dist:.4f}"
    )

    # Nearest-neighbor redundancy (how many are extremely close to another)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(emb)
    distances, indices_nn = nn.kneighbors(emb)
    # distances[:, 0] is distance to itself; take the second closest
    nn_dists = distances[:, 1]
    threshold = 0.05  # very small cosine distance -> high redundancy
    redundancy_rate = float(np.mean(nn_dists < threshold))

    print(
        f"[SBERT] Redundancy rate (nearest cosine distance < {threshold}): "
        f"{redundancy_rate:.4f}"
    )

    # PCA visualization

    
    print("[SBERT] Computing PCA projection for visualization...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)

    # For plotting, optionally sample down if very large
    plot_n = min(2000, n)
    plot_idx = np.random.choice(n, size=plot_n, replace=False)
    x = emb_2d[plot_idx, 0]
    y = emb_2d[plot_idx, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=5, alpha=0.4)

    # Axis labels: bigger + bold
    plt.xlabel("PC1", fontsize=16, fontweight="bold")
    plt.ylabel("PC2", fontsize=16, fontweight="bold")

    # Optional: make tick labels bigger (recommended for readability)
    plt.tick_params(axis="both", which="major", labelsize=12)

    ensure_fig_dir()
    out_path = os.path.join(FIG_DIR, "panopticon_sbert_pca.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SBERT] PCA figure saved to: {out_path}")

    return {
        "avg_pairwise_cosine_distance": avg_pairwise_dist,
        "redundancy_rate_cosine<0.05": redundancy_rate,
        "num_samples": int(n),
    }


def generate_prompt_wordcloud(dataset, max_prompts=20000):
    print("[WordCloud] Generating prompt word cloud...")
    _ensure_nltk_stopwords()
    sw = set(stopwords.words("english"))

    total = len(dataset)
    limit = min(total, max_prompts)
    texts = [dataset[i]["Prompt"] for i in range(limit)]

    # Join and lowercase
    text = " ".join(texts).lower()

    # Basic cleanup: you can add more domain-specific filtering here
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        stopwords=sw,
        collocations=False,
    ).generate(text)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    ensure_fig_dir()
    out_path = os.path.join(FIG_DIR, "panopticon_prompt_wordcloud.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[WordCloud] Figure saved to: {out_path}")

def check_pii_leakage(dataset):
    leak = []
    for d in dataset:
        if d['Category'] == 'Benign':
            pii = d['PromptPII']
            leaked_pii = {k: v for k, v in pii.items() if v is not None}
            if leaked_pii:
                leak.append(leaked_pii)
    return leak


if __name__ == "__main__":



    '''leaks = check_pii_leakage(dataset)
    print(f'numm leaks {len(leaks)}')
    print('------------')
    print(leaks)'''

    ensure_fig_dir()

    # 1. Duplicate analysis
    uniques, duplicates, duplicate_dict, categories_dict = dataset_diversity(dataset)

    # 2. PII label distribution
    labels = handle_pii_labels(dataset)

    # 3. Lexical diversity metrics
    lexical_stats = lexical_diversity_metrics(dataset, max_prompts=100_000)

    # 4. SBERT-based diversity metrics
    sbert_stats = sbert_diversity_metrics(
        dataset,
        sample_size=len(dataset),  # adjust up/down depending on compute
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    # 5. Prompt word cloud
    generate_prompt_wordcloud(dataset, max_prompts=100_000)

    print("\n=== SUMMARY METRICS ===")
    print("Lexical:", lexical_stats)
    print("SBERT:", sbert_stats)
