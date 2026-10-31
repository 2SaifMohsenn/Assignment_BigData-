"""Reproducible SMS spam analysis used by notebook/analysis.ipynb."""

from collections import Counter
from pathlib import Path
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def safe_show():
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42


def section(title):
    print(f"\n{'=' * 14} {title} {'=' * 14}")


def clean_message(value):
    """Normalize SMS text without changing the source text column."""
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def top_words(messages, limit=10, remove_stopwords=True):
    tokens = []
    for message in messages:
        words = message.split()
        if remove_stopwords:
            words = [word for word in words if word not in ENGLISH_STOP_WORDS]
        tokens.extend(words)
    return Counter(tokens).most_common(limit)


def distribution_table(samples, labels):
    distributions = {"Original": labels.value_counts(normalize=True)}
    for name, sample in samples.items():
        distributions[name] = sample["label"].value_counts(normalize=True)
    result = pd.DataFrame(distributions).fillna(0).T
    result.index.name = "sample"
    return result.reindex(columns=sorted(labels.unique())).fillna(0)


def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    required = {"label", "text"}
    if not required.issubset(raw.columns):
        raise ValueError(f"{DATA_PATH} must contain columns {sorted(required)}")

    section("Dataset understanding")
    print("Source shape:", raw.shape)
    print("\nHead:\n", raw.head().to_string(index=False))
    print("\nColumns:", raw.columns.tolist())
    print("\nDtypes:\n", raw.dtypes.to_string())
    print("\nInfo:")
    raw.info()
    print("\nDescriptive statistics:\n", raw.describe(include="all").to_string())
    print("\nMissing values per column:\n", raw.isna().sum().to_string())
    duplicate_count = int(raw.duplicated(subset=["label", "text"]).sum())
    print("\nExact duplicate label/text rows:", duplicate_count)

    section("Cleaning and transformations")
    before_shape = raw.shape
    data = raw.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["text"] = data["text"].fillna("").astype(str)
    data = data.loc[data["label"].isin(["ham", "spam"])].copy()
    data = data.drop_duplicates(subset=["label", "text"], keep="first").copy()
    data["cleaned_text"] = data["text"].map(clean_message)
    empty_text_count = int(data["cleaned_text"].eq("").sum())
    data = data.loc[data["cleaned_text"].ne("")].copy()
    data.reset_index(drop=True, inplace=True)
    print("Shape before cleaning:", before_shape)
    print("Shape after cleaning:", data.shape)
    print(
        "Decision: remove exact duplicate labeled messages to avoid repeated records "
        "overweighting the results; remove unusable empty messages; preserve original "
        "text and fill missing message text before normalization."
    )
    print("Empty normalized messages removed:", empty_text_count)
    print("Original text preserved:", bool(data["text"].notna().all()))

    data["tokens"] = data["cleaned_text"].str.split()
    data["char_count"] = data["text"].str.len()
    data["word_count"] = data["tokens"].str.len()
    data["unique_word_count"] = data["tokens"].map(lambda words: len(set(words)))
    data["avg_word_length"] = data.apply(
        lambda row: (
            float(np.mean([len(word) for word in row["tokens"]]))
            if row["tokens"]
            else 0.0
        ),
        axis=1,
    )
    data["digit_count"] = data["text"].str.count(r"\d")
    data["label_code"] = data["label"].map({"ham": 0, "spam": 1}).astype("int8")
    char_range = data["char_count"].max() - data["char_count"].min()
    data["char_count_normalized"] = (
        (data["char_count"] - data["char_count"].min()) / char_range
        if char_range
        else 0.0
    )
    data["length_bin"] = pd.cut(
        data["word_count"],
        bins=[-1, 8, 20, np.inf],
        labels=["short", "medium", "long"],
    )
    print("\nCreated transformations: cleaned_text, label_code, character-count "
          "normalization, word-count bins, and numeric text features.")
    print("\nNumerical feature summary:\n", data[
        ["char_count", "word_count", "unique_word_count", "avg_word_length", "digit_count"]
    ].describe().round(2).to_string())

    section("loc and iloc examples")
    print("\nloc: spam messages with original text (first five):\n", data.loc[
        data["label"].eq("spam"), ["label", "text"]
    ].head().to_string(index=False))
    positions = np.r_[0:5, max(0, len(data) - 5):len(data)]
    print("\niloc with np.r_: first and last five records:\n", data.iloc[
        positions, [0, 1, 2]
    ].to_string(index=False))

    section("Sampling comparison")
    original_labels = data["label"].copy()
    simple_random = data.sample(frac=0.20, random_state=RANDOM_STATE)
    stratified = train_test_split(
        data,
        train_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )[0]
    data["message_length_quartile"] = pd.qcut(
        data["word_count"], q=4, duplicates="drop"
    )
    quartiles = list(data["message_length_quartile"].dropna().cat.categories)
    if not quartiles:
        raise ValueError("Cannot create message-length groups for cluster sampling")
    selected_quartile = quartiles[int(np.random.default_rng(RANDOM_STATE).integers(len(quartiles)))]
    cluster_sample = data.loc[data["message_length_quartile"].eq(selected_quartile)].copy()
    samples = {
        "Simple random": simple_random,
        "Stratified": stratified,
        f"Length cluster {selected_quartile}": cluster_sample,
    }
    sampling_distributions = distribution_table(samples, original_labels)
    print("Random sample selects rows uniformly; stratified sampling preserves label "
          "proportions; this illustrative cluster sample takes all records from one "
          "message-length quartile and can therefore shift label proportions.")
    print("\nClass proportions (fractions):\n", sampling_distributions.round(3).to_string())
    print("\nSample sizes:\n", pd.Series({name: len(sample) for name, sample in samples.items()}).to_string())
    sampling_distributions.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    plt.ylabel("Fraction of records")
    plt.title("Original and sampled label distributions")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sampling_distributions.png", dpi=160)
    safe_show()
    plt.close()

    section("Text analysis")
    token_lists = data["tokens"].tolist()
    total_tokens = sum(map(len, token_lists))
    unique_words = len({word for words in token_lists for word in words})
    average_words = float(data["word_count"].mean())
    print("Total tokens:", total_tokens)
    print("Unique words:", unique_words)
    print(f"Average words per record: {average_words:.2f}")

    frequent_all = top_words(data["cleaned_text"], limit=20, remove_stopwords=False)
    frequent_no_stop = top_words(data["cleaned_text"], limit=20, remove_stopwords=True)
    frequency_table = pd.DataFrame(
        {
            "rank": np.arange(1, 21),
            "including_stopwords": [word for word, _ in frequent_all],
            "count_including_stopwords": [count for _, count in frequent_all],
            "excluding_stopwords": [word for word, _ in frequent_no_stop],
            "count_excluding_stopwords": [count for _, count in frequent_no_stop],
        }
    )
    print("\nTop 20 words, before and after English stopword removal:\n",
          frequency_table.to_string(index=False))
    frequency_table.to_csv(OUTPUT_DIR / "top_words.csv", index=False)
    word_plot = pd.DataFrame(frequent_no_stop, columns=["word", "count"])
    sns.barplot(data=word_plot, x="count", y="word", color="#4472C4")
    plt.title("Top 20 SMS words after stopword removal")
    plt.xlabel("Token frequency")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "word_frequency_top20.png", dpi=160)
    safe_show()
    plt.close()

    group_stats = data.groupby("label", observed=True).agg(
        average_characters=("char_count", "mean"),
        average_words=("word_count", "mean"),
    )
    print("\nAverage length comparison:\n", group_stats.round(2).to_string())
    for label in sorted(data["label"].unique()):
        print(f"\nTop 10 words for {label} (stopwords removed): "
              f"{top_words(data.loc[data['label'].eq(label), 'cleaned_text'], 10)}")
    group_stats.plot(kind="bar", figsize=(8, 5))
    plt.title("Average message length by label")
    plt.ylabel("Average count")
    plt.xlabel("Message label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "group_comparison.png", dpi=160)
    safe_show()
    plt.close()

    section("Correlation")
    numeric_features = [
        "char_count",
        "word_count",
        "unique_word_count",
        "avg_word_length",
        "digit_count",
    ]
    correlation = data[numeric_features].corr()
    print(correlation.round(3).to_string())
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation among numerical text features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    safe_show()
    plt.close()
    print("Correlations show association, not causation; message length and token "
          "counts are mechanically related, but correlation alone cannot establish why.")

    section("Supervised TF-IDF classification")
    X_train, X_test, y_train, y_test = train_test_split(
        data["cleaned_text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    classifier_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True
    )
    X_train_tfidf = classifier_vectorizer.fit_transform(X_train)
    X_test_tfidf = classifier_vectorizer.transform(X_test)
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    model_metrics = {}
    predictions = {}
    for name, model in classifiers.items():
        model.fit(X_train_tfidf, y_train)
        predicted = model.predict(X_test_tfidf)
        predictions[name] = predicted
        model_metrics[name] = {
            "accuracy": accuracy_score(y_test, predicted),
            "precision_spam": precision_score(y_test, predicted, pos_label="spam", zero_division=0),
            "recall_spam": recall_score(y_test, predicted, pos_label="spam", zero_division=0),
            "f1_spam": f1_score(y_test, predicted, pos_label="spam", zero_division=0),
        }
        print(f"\n{name}:\n", pd.Series(model_metrics[name]).round(4).to_string())
        print(classification_report(y_test, predicted, labels=["ham", "spam"], zero_division=0))

    metrics_lines = [
        "SMS Spam Collection — supervised TF-IDF model results",
        f"Records after cleaning: {len(data)}",
        f"Train/test sizes: {len(X_train)}/{len(X_test)} (stratified 80/20 split)",
        f"TF-IDF vocabulary size: {len(classifier_vectorizer.vocabulary_)}",
        "Positive class for precision, recall and F1: spam",
        "Limitation: this is a small, historically collected corpus; message topics "
        "and writing habits may differ from modern SMS, and class imbalance can make "
        "accuracy alone misleading.",
    ]
    for name, values in model_metrics.items():
        metrics_lines.extend(
            [
                "",
                name,
                *(f"{metric}: {value:.4f}" for metric, value in values.items()),
                classification_report(
                    y_test, predictions[name], labels=["ham", "spam"], zero_division=0
                ),
            ]
        )
    (OUTPUT_DIR / "model_metrics.txt").write_text("\n".join(metrics_lines), encoding="utf-8")

    fig, axes = plt.subplots(1, len(classifiers), figsize=(11, 4))
    for axis, (name, predicted) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_test, predicted, labels=["ham", "spam"])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
            ax=axis,
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
    fig.suptitle("Test-set confusion matrices")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_confusion_matrices.png", dpi=160)
    safe_show()
    plt.close(fig)

    section("K-Means clustering")
    cluster_vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=5000
    )
    X_cluster = cluster_vectorizer.fit_transform(data["cleaned_text"])
    sample_size = min(1000, X_cluster.shape[0])
    model_candidates = {}
    score_by_k = {}
    for k in (2, 3, 4):
        if k >= X_cluster.shape[0]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            cluster_labels = model.fit_predict(X_cluster)
        score = silhouette_score(
            X_cluster,
            cluster_labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
            metric="cosine",
        )
        model_candidates[k] = (model, cluster_labels)
        score_by_k[k] = float(score)
        print(f"k={k}: sampled cosine silhouette={score:.4f}")
    if not score_by_k:
        raise ValueError("Not enough records to evaluate K-Means candidates")

    selected_k = max(score_by_k, key=score_by_k.get)
    selected_model, cluster_labels = model_candidates[selected_k]
    data["cluster"] = cluster_labels
    cluster_sizes = data["cluster"].value_counts().sort_index()
    print(f"\nSelected k={selected_k} by the highest sampled cosine silhouette score.")
    print("Cluster sizes:\n", cluster_sizes.to_string())
    cluster_terms = []
    feature_names = cluster_vectorizer.get_feature_names_out()
    summary_lines = [
        "SMS Spam Collection — K-Means clustering summary",
        f"Representation: {X_cluster.shape[1]}-feature TF-IDF, word uni-/bigrams",
        f"Candidate k values and sampled cosine silhouette scores: {score_by_k}",
        f"Selected k: {selected_k}",
        f"Selected-model sampled cosine silhouette score: {score_by_k[selected_k]:.4f}",
        "Silhouette was sampled for computational practicality; it measures separation "
        "in the TF-IDF representation, not agreement with spam labels.",
        "",
        "Cluster interpretation (terms and majority known label are descriptive only):",
    ]
    label_cross_tab = pd.crosstab(data["cluster"], data["label"], normalize="index")
    for cluster_id in sorted(cluster_sizes.index):
        member_rows = np.flatnonzero(cluster_labels == cluster_id)
        mean_weights = np.asarray(X_cluster[member_rows].mean(axis=0)).ravel()
        top_indices = mean_weights.argsort()[-10:][::-1]
        terms = [feature_names[index] for index in top_indices if mean_weights[index] > 0]
        cluster_terms.append(
            {"cluster": int(cluster_id), "size": int(cluster_sizes.loc[cluster_id]), "top_terms": ", ".join(terms)}
        )
        label_mix = label_cross_tab.loc[cluster_id].sort_values(ascending=False)
        majority_label = str(label_mix.index[0])
        summary_lines.append(
            f"Cluster {cluster_id}: n={cluster_sizes.loc[cluster_id]}; "
            f"top terms={', '.join(terms)}; label proportions={label_mix.round(3).to_dict()}. "
            f"Descriptive majority label: {majority_label}."
        )
    summary_lines.extend(
        [
            "",
            "Compare clusters with labels using proportions, not cluster-ID equality. "
            "A cluster may mix ham and spam because K-Means groups text similarity "
            "without access to the labels.",
            "\nCluster/label counts:\n" + pd.crosstab(data["cluster"], data["label"]).to_string(),
        ]
    )
    (OUTPUT_DIR / "cluster_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    pd.DataFrame(cluster_terms).to_csv(OUTPUT_DIR / "cluster_terms.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=cluster_sizes.index.astype(str), y=cluster_sizes.values, color="#70AD47")
    plt.title(f"K-Means cluster sizes (selected k={selected_k})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of messages")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_cluster_sizes.png", dpi=160)
    safe_show()
    plt.close()

    section("Evidence-based insights")
    label_proportions = data["label"].value_counts(normalize=True)
    ham_words = top_words(data.loc[data["label"].eq("ham"), "cleaned_text"], 1)
    spam_words = top_words(data.loc[data["label"].eq("spam"), "cleaned_text"], 1)
    best_model_name = max(model_metrics, key=lambda name: model_metrics[name]["f1_spam"])
    best_metrics = model_metrics[best_model_name]
    ham_chars = group_stats.loc["ham", "average_characters"]
    spam_chars = group_stats.loc["spam", "average_characters"]
    if ham_chars:
        length_ratio = spam_chars / ham_chars
    else:
        length_ratio = float("nan")
    insights = [
        (
            f"Observation: ham accounts for {label_proportions.get('ham', 0):.1%} "
            f"of cleaned records. Evidence: {int((data['label'] == 'ham').sum())} "
            f"ham and {int((data['label'] == 'spam').sum())} spam messages remain. "
            "Interpretation: the class imbalance makes per-class recall/F1 important "
            "alongside accuracy."
        ),
        (
            f"Observation: spam messages average {spam_chars:.1f} characters versus "
            f"{ham_chars:.1f} for ham (ratio {length_ratio:.2f}). Evidence: grouped "
            "character-count means shown in the group-comparison output. "
            "Interpretation: message length differs by class in this corpus, but length "
            "alone is not a reliable decision rule."
        ),
        (
            f"Observation: the most frequent stopword-filtered ham token is "
            f"'{ham_words[0][0]}' ({ham_words[0][1]} occurrences), while the spam "
            f"leader is '{spam_words[0][0]}' ({spam_words[0][1]} occurrences). "
            "Evidence: per-label top-word counts. Interpretation: class-specific "
            "vocabulary provides useful but corpus-dependent signal."
        ),
        (
            f"Observation: {best_model_name} achieved spam F1={best_metrics['f1_spam']:.3f} "
            f"and spam recall={best_metrics['recall_spam']:.3f} on the stratified test "
            "set. Evidence: held-out classification metrics. Interpretation: this "
            "balances false positives and missed spam differently from accuracy alone."
        ),
        (
            f"Observation: the selected K-Means solution has k={selected_k}, sampled "
            f"cosine silhouette={score_by_k[selected_k]:.3f}. Evidence: candidate-k "
            "silhouette experiment and cluster/label proportions. Interpretation: "
            "clusters summarize TF-IDF similarity and can contain mixed known labels."
        ),
        (
            "Observation: character count and word count both describe message size. "
            "Evidence: the correlation heatmap reports their measured association. "
            "Interpretation: overlapping length measures are expected, but correlation "
            "is descriptive and cannot establish causality."
        ),
    ]
    for number, insight in enumerate(insights, start=1):
        print(f"{number}. {insight}")
    (OUTPUT_DIR / "insights.txt").write_text(
        "\n".join(f"{number}. {insight}" for number, insight in enumerate(insights, start=1)),
        encoding="utf-8",
    )

    data.drop(columns=["tokens", "message_length_quartile"]).to_csv(
        OUTPUT_DIR / "cleaned_data.csv", index=False
    )
    print(f"\nCleaned data saved to {OUTPUT_DIR / 'cleaned_data.csv'}")
    print(f"Reports and figures saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    run_analysis()
