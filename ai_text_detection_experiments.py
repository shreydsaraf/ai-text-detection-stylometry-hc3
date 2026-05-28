"""
AI vs human text detection experiments with stylometric ablation.

This script is designed for the SIT330 2.3HD submission update. It avoids
notebook screenshots by saving results as CSV tables and publication-ready
figures that can be inserted into the report.

Default public benchmark:
    HC3 open_qa split from Hello-SimpleAI/HC3
    https://huggingface.co/datasets/Hello-SimpleAI/HC3

Example:
    python ai_text_detection_experiments.py --dataset hc3

Custom CSV example:
    python ai_text_detection_experiments.py \
        --dataset csv \
        --data /path/to/AI_Human.csv \
        --text-column text \
        --label-column generated \
        --output-dir results_public

The local synthetic dataset should not be used for final claims. It is kept
only as an optional debugging input:
    python ai_text_detection_experiments.py \
        --dataset csv \
        --data /Users/sakshisaraf/Documents/SIT330/final_hd_dataset.csv \
        --text-column text \
        --label-column label \
        --output-dir results_synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
LABEL_NAMES = {0: "Human", 1: "AI"}
HC3_OPEN_QA_URL = (
    "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl"
)


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: list[str]


def normalise_label(value: object) -> int:
    """Map common human/AI label formats onto 0=human and 1=AI."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"0", "human", "real", "student", "original"}:
            return 0
        if cleaned in {"1", "ai", "generated", "machine", "chatgpt", "llm"}:
            return 1
    numeric = int(float(value))
    if numeric not in {0, 1}:
        raise ValueError(f"Unsupported binary label: {value!r}")
    return numeric


def load_dataset(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Text column {text_column!r} not found in {path}")
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column!r} not found in {path}")

    out = df[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    out = out.dropna(subset=["text", "label"]).copy()
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].drop_duplicates(subset=["text"])
    out["label"] = out["label"].map(normalise_label)
    return out.reset_index(drop=True)


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public dataset from {url}")
    urllib.request.urlretrieve(url, path)


def load_hc3(path: Path, max_answers_per_question: int = 1) -> pd.DataFrame:
    """Load HC3 JSONL into text,label rows.

    HC3 records contain paired human_answers and chatgpt_answers for the same
    question. Each human answer becomes label 0 and each ChatGPT answer becomes
    label 1. By default, one answer from each side is used per question to keep
    the dataset balanced and avoid overweighting questions with many answers.
    """
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for question_id, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            for answer in item.get("human_answers", [])[:max_answers_per_question]:
                if str(answer).strip():
                    rows.append(
                        {
                            "question_id": question_id,
                            "text": str(answer).strip(),
                            "label": 0,
                        }
                    )
            for answer in item.get("chatgpt_answers", [])[:max_answers_per_question]:
                if str(answer).strip():
                    rows.append(
                        {
                            "question_id": question_id,
                            "text": str(answer).strip(),
                            "label": 1,
                        }
                    )

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_experiment_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset == "hc3":
        data_path = (
            Path(args.data)
            if args.data
            else Path(args.download_dir) / "hc3_open_qa.jsonl"
        )
        if not data_path.exists():
            download_file(HC3_OPEN_QA_URL, data_path)
        args.resolved_data_path = str(data_path)
        return load_hc3(data_path, args.max_answers_per_question)

    if not args.data:
        raise ValueError("--data is required when --dataset csv is used.")
    args.resolved_data_path = args.data
    return load_dataset(Path(args.data), args.text_column, args.label_column)


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def extract_stylometric_features(text: str) -> dict[str, float]:
    text = str(text)
    chars = list(text)
    words = re.findall(r"\b[\w']+\b", text)
    lower_words = [word.lower() for word in words]
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    word_count = len(words)
    char_count = len(chars)
    sentence_count = max(len(sentences), 1)
    unique_words = len(set(lower_words))
    syllables = sum(count_syllables(word) for word in words)
    stopwords = sum(1 for word in lower_words if word in ENGLISH_STOP_WORDS)
    punctuation = sum(1 for char in chars if char in string.punctuation)
    digits = sum(1 for char in chars if char.isdigit())
    uppercase = sum(1 for char in chars if char.isupper())
    commas = text.count(",")
    periods = text.count(".")
    exclamations = text.count("!")
    questions = text.count("?")
    quotes = text.count('"') + text.count("'")

    avg_sentence_length = safe_divide(word_count, sentence_count)
    avg_word_length = safe_divide(sum(len(word) for word in words), word_count)
    flesch = (
        206.835
        - 1.015 * avg_sentence_length
        - 84.6 * safe_divide(syllables, word_count)
        if word_count
        else 0.0
    )

    return {
        "surface_word_count": float(word_count),
        "surface_char_count": float(char_count),
        "surface_sentence_count": float(sentence_count),
        "surface_avg_word_length": avg_word_length,
        "surface_avg_sentence_length": avg_sentence_length,
        "lexical_diversity": safe_divide(unique_words, word_count),
        "lexical_stopword_ratio": safe_divide(stopwords, word_count),
        "lexical_long_word_ratio": safe_divide(
            sum(1 for word in words if len(word) >= 7), word_count
        ),
        "punct_punctuation_ratio": safe_divide(punctuation, char_count),
        "punct_comma_ratio": safe_divide(commas, word_count),
        "punct_period_ratio": safe_divide(periods, word_count),
        "punct_exclamation_ratio": safe_divide(exclamations, word_count),
        "punct_question_ratio": safe_divide(questions, word_count),
        "punct_quote_ratio": safe_divide(quotes, word_count),
        "orth_uppercase_ratio": safe_divide(uppercase, char_count),
        "orth_digit_ratio": safe_divide(digits, char_count),
        "readability_syllables_per_word": safe_divide(syllables, word_count),
        "readability_flesch_reading_ease": flesch,
    }


def build_feature_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_stylometric_features(text) for text in texts])


def feature_groups(columns: list[str]) -> list[FeatureGroup]:
    groups = []
    for prefix, label in [
        ("surface_", "surface_length"),
        ("lexical_", "lexical_choice"),
        ("punct_", "punctuation"),
        ("orth_", "orthography"),
        ("readability_", "readability"),
    ]:
        selected = [col for col in columns if col.startswith(prefix)]
        if selected:
            groups.append(FeatureGroup(label, selected))
    groups.append(FeatureGroup("all_stylometric", columns))
    return groups


def metric_row(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    feature_set: str,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model": name,
        "feature_set": feature_set,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        row["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        row["roc_auc"] = np.nan
    return row


def decision_scores(model: object, x_test: object) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return np.asarray(scores)
    return None


def train_text_models(
    x_train_text: pd.Series,
    x_test_text: pd.Series,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    models: dict[str, object] = {
        "TF-IDF + Logistic Regression": LogisticRegression(max_iter=2000),
        "TF-IDF + Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "TF-IDF + Complement Naive Bayes": ComplementNB(),
    }
    rows = []
    predictions = {}
    for name, estimator in models.items():
        pipeline = make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
            ),
            estimator,
        )
        pipeline.fit(x_train_text, y_train)
        y_pred = pipeline.predict(x_test_text)
        scores = decision_scores(pipeline, x_test_text)
        rows.append(metric_row(name, y_test, y_pred, scores, "tfidf_only"))
        predictions[name] = y_pred
    return pd.DataFrame(rows), predictions


def train_stylometric_models(
    x_train_style: pd.DataFrame,
    x_test_style: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions = {}
    groups = feature_groups(list(x_train_style.columns))
    for group in groups:
        train_part = x_train_style[group.columns]
        test_part = x_test_style[group.columns]
        candidates: dict[str, object] = {
            "Style + Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000)
            ),
            "Style + Random Forest": RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                min_samples_leaf=2,
            ),
        }
        for name, model in candidates.items():
            model.fit(train_part, y_train)
            y_pred = model.predict(test_part)
            scores = decision_scores(model, test_part)
            feature_set = group.name
            rows.append(metric_row(name, y_test, y_pred, scores, feature_set))
            predictions[f"{name} ({feature_set})"] = y_pred
    return pd.DataFrame(rows), predictions


def train_hybrid_model(
    x_train_text: pd.Series,
    x_test_text: pd.Series,
    x_train_style: pd.DataFrame,
    x_test_style: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
    )
    scaler = MinMaxScaler()
    x_train_tfidf = vectorizer.fit_transform(x_train_text)
    x_test_tfidf = vectorizer.transform(x_test_text)
    x_train_scaled = scaler.fit_transform(x_train_style)
    x_test_scaled = scaler.transform(x_test_style)
    x_train_all = hstack([x_train_tfidf, csr_matrix(x_train_scaled)])
    x_test_all = hstack([x_test_tfidf, csr_matrix(x_test_scaled)])

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train_all, y_train)
    y_pred = model.predict(x_test_all)
    scores = decision_scores(model, x_test_all)
    rows = [
        metric_row(
            "TF-IDF + all stylometric + Logistic Regression",
            y_test,
            y_pred,
            scores,
            "hybrid_all",
        )
    ]
    return pd.DataFrame(rows), {"Hybrid Logistic Regression": y_pred}


def save_bar_chart(results: pd.DataFrame, output_path: Path) -> None:
    ranked = results.sort_values("f1", ascending=False).head(12).copy()
    labels = ranked["model"] + "\n" + ranked["feature_set"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, ranked["f1"], color="#3f7cac")
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Model comparison by F1 score")
    for idx, value in enumerate(ranked["f1"]):
        ax.text(min(value + 0.01, 0.98), idx, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_ablation_chart(results: pd.DataFrame, output_path: Path) -> None:
    ablation = results[
        results["model"].str.contains("Style", regex=False)
        & (results["feature_set"] != "all_stylometric")
    ].copy()
    if ablation.empty:
        return
    pivot = ablation.pivot_table(
        index="feature_set", columns="model", values="f1", aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax, color=["#c65d4e", "#4e8f68"])
    ax.set_xlabel("Stylometric feature group")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Stylometric ablation study")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Best model confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_dataset_profile(df: pd.DataFrame, output_dir: Path) -> None:
    profile = pd.DataFrame(
        {
            "metric": [
                "samples",
                "human_samples",
                "ai_samples",
                "duplicates_removed",
                "mean_words",
                "median_words",
            ],
            "value": [
                len(df),
                int((df["label"] == 0).sum()),
                int((df["label"] == 1).sum()),
                0,
                float(df["text"].str.split().str.len().mean()),
                float(df["text"].str.split().str.len().median()),
            ],
        }
    )
    profile.to_csv(output_dir / "dataset_profile.csv", index=False)


def save_error_analysis(
    x_test_text: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    errors = pd.DataFrame(
        {
            "text": x_test_text.reset_index(drop=True),
            "true_label": [LABEL_NAMES[int(label)] for label in y_test],
            "predicted_label": [LABEL_NAMES[int(label)] for label in y_pred],
        }
    )
    errors = errors[errors["true_label"] != errors["predicted_label"]]
    errors.head(25).to_csv(output_path, index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiment_dataset(args)
    if args.max_samples and len(df) > args.max_samples:
        df = (
            df.groupby("label", group_keys=False)
            .sample(
                n=max(1, math.floor(args.max_samples / 2)),
                random_state=RANDOM_STATE,
            )
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    style = build_feature_frame(df["text"])
    x_train_text, x_test_text, y_train, y_test, x_train_style, x_test_style = (
        train_test_split(
            df["text"],
            df["label"].to_numpy(),
            style,
            test_size=args.test_size,
            random_state=RANDOM_STATE,
            stratify=df["label"],
        )
    )

    text_results, text_predictions = train_text_models(
        x_train_text, x_test_text, y_train, y_test
    )
    style_results, style_predictions = train_stylometric_models(
        x_train_style, x_test_style, y_train, y_test
    )
    hybrid_results, hybrid_predictions = train_hybrid_model(
        x_train_text,
        x_test_text,
        x_train_style,
        x_test_style,
        y_train,
        y_test,
    )

    results = pd.concat(
        [text_results, style_results, hybrid_results], ignore_index=True
    ).sort_values(["f1", "accuracy"], ascending=False)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    save_dataset_profile(df, output_dir)
    save_bar_chart(results, output_dir / "model_comparison.png")
    save_ablation_chart(results, output_dir / "stylometric_ablation.png")

    best_name = str(results.iloc[0]["model"])
    all_predictions = {
        **text_predictions,
        **style_predictions,
        **hybrid_predictions,
    }
    matching_key = next(
        key for key in all_predictions if key.startswith(best_name.split(" (")[0])
    )
    best_pred = all_predictions[matching_key]
    save_confusion(y_test, best_pred, output_dir / "best_confusion_matrix.png")
    save_error_analysis(
        x_test_text,
        y_test,
        best_pred,
        output_dir / "misclassified_examples.csv",
    )

    report = classification_report(
        y_test,
        best_pred,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data": str(Path(getattr(args, "resolved_data_path", args.data)).resolve()),
                "samples_after_cleaning": len(df),
                "test_size": args.test_size,
                "best_model": best_name,
                "best_feature_set": str(results.iloc[0]["feature_set"]),
                "best_f1": float(results.iloc[0]["f1"]),
                "classification_report": report,
            },
            f,
            indent=2,
        )

    print("\nTop results")
    print(results.head(10).to_string(index=False))
    print(f"\nSaved tables and figures to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI-vs-human text detection experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["hc3", "csv"],
        default="hc3",
        help="Dataset source. Use hc3 for the public HC3 benchmark or csv for a custom file.",
    )
    parser.add_argument(
        "--data",
        default="",
        help="Path to a CSV file, or an existing HC3 JSONL file when --dataset hc3.",
    )
    parser.add_argument(
        "--download-dir",
        default="data_public",
        help="Directory used for downloading public datasets.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", default="results_hc3")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-answers-per-question", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional balanced sample cap for quick public-dataset runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
