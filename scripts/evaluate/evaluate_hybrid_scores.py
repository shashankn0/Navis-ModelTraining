"""Hybrid evaluation: combine human ratings with automatic metrics and rank model x prompt pairs.

Inputs (defaults are relative to the repo root, whatever directory you run this from):
    results/benchmark/benchmark_results.csv   benchmark output with sentiment_score and response_length
    results/human_rating/human_ratings.csv    customer_query, model_response, human_rating (1-5)

Outputs (in --out-dir, default results/evaluation):
    merged_full_evaluation.csv   one row per rated response, with normalised metrics and hybrid_score
    final_prompt_ranking.csv     mean metrics per (model, prompt), best first

Scoring profiles (--profile):
    v1  The original formula, unchanged and the default, so published results reproduce exactly:
            0.55 human + 0.20 sentiment + 0.20 speed + 0.05 brevity
        Every metric is min-max scaled over the values present in the data. Note this means a rating
        scale that only reaches 2-5 maps a "2" to 0.0, not 0.25.
    v2  EXPERIMENTAL, not yet validated against anything:
            0.60 human + 0.15 reference_similarity + 0.20 speed + 0.05 brevity   (sentiment weight 0)
        Human ratings use the fixed 1-5 scale. Sentiment is dropped because the prompts themselves ask
        for a warm or a formal tone, so sentiment mostly measures the instruction, not the answer.
        reference_similarity is the TF-IDF cosine similarity to the dataset's expected_response: a
        crude lexical proxy, but a use of reference answers the benchmark already has.

reference_similarity is always computed and reported, even under v1, where it carries no weight.

Run:  python scripts/evaluate/evaluate_hybrid_scores.py [--profile v1|v2] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO_ROOT / "results" / "benchmark" / "benchmark_results.csv"
DEFAULT_HUMAN = REPO_ROOT / "results" / "human_rating" / "human_ratings.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "evaluation"

MERGE_KEYS = ["customer_query", "model_response"]

# Weights per profile. "speed" and "brevity" are inverted min-max scores (lower latency / shorter is better).
PROFILES: dict[str, dict[str, float]] = {
    "v1": {"human": 0.55, "sentiment": 0.20, "speed": 0.20, "brevity": 0.05, "reference": 0.0},
    "v2": {"human": 0.60, "sentiment": 0.00, "speed": 0.20, "brevity": 0.05, "reference": 0.15},
}


def minmax(series: pd.Series) -> pd.Series:
    """Scale to [0, 1] over the observed values. A constant column becomes all zeros (as scikit-learn's MinMaxScaler does)."""
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(0.0, index=series.index).where(series.notna())
    return (series - lo) / (hi - lo)


def merge_ratings(bench: pd.DataFrame, human: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach human_rating to benchmark rows, matching on (customer_query, model_response).

    Raises ValueError rather than silently mis-joining: duplicated keys would multiply rows, and a join
    that matches nothing means the two files are out of sync.
    Returns the merged frame and counts of rows that could not be matched.
    """
    for col in MERGE_KEYS + ["human_rating"]:
        if col not in human.columns:
            raise ValueError(f"human ratings file is missing the '{col}' column")
    for col in MERGE_KEYS:
        if col not in bench.columns:
            raise ValueError(f"benchmark results file is missing the '{col}' column")

    bench = bench.dropna(subset=["model_response"])  # rows where the API call failed
    human = human.dropna(subset=["human_rating"])  # responses not rated yet

    for name, frame in (("benchmark results", bench), ("human ratings", human)):
        dupes = frame.duplicated(MERGE_KEYS, keep=False)
        if dupes.any():
            raise ValueError(
                f"{int(dupes.sum())} rows in the {name} share the same (customer_query, model_response); "
                "they cannot be matched one-to-one. Remove or de-duplicate them."
            )

    merged = bench.merge(human[MERGE_KEYS + ["human_rating"]], on=MERGE_KEYS, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no benchmark row matched a human rating; are these files from the same run?")
    return merged, {
        "benchmark_unmatched": len(bench) - len(merged),
        "ratings_unmatched": len(human) - len(merged),
    }


def reference_similarity(responses: pd.Series, references: pd.Series) -> pd.Series:
    """TF-IDF cosine similarity between each response and its reference answer (NaN if either is empty)."""
    responses = responses.fillna("").astype(str)
    references = references.fillna("").astype(str)
    valid = (responses.str.strip() != "") & (references.str.strip() != "")
    result = pd.Series(np.nan, index=responses.index)
    if not valid.any():
        return result

    try:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english").fit(
            pd.concat([responses[valid], references[valid]])
        )
    except ValueError:  # every text is made only of stop words: nothing to compare
        return result
    a = vectorizer.transform(responses[valid])
    b = vectorizer.transform(references[valid])
    # TF-IDF rows are L2-normalised, so the row-wise dot product is the cosine similarity.
    result.loc[valid] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return result


def add_scores(df: pd.DataFrame, profile: str = "v1") -> pd.DataFrame:
    """Return a copy of df with normalised metrics and hybrid_score for the chosen profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile '{profile}' (choose from {', '.join(PROFILES)})")
    weights = PROFILES[profile]
    out = df.copy()

    if profile == "v1":
        out["human_norm"] = minmax(out["human_rating"])
        out["sentiment_norm_final"] = minmax(out["sentiment_score"])
    else:
        out["human_norm"] = (out["human_rating"] - 1.0) / 4.0
        out["sentiment_norm_final"] = (out["sentiment_score"] + 1.0) / 2.0

    out["latency_norm"] = minmax(out["latency_seconds"])
    out["length_norm"] = minmax(out["response_length"])
    out["reference_similarity"] = reference_similarity(out["model_response"], out["expected_response"])

    out["hybrid_score"] = (
        weights["human"] * out["human_norm"]
        + weights["sentiment"] * out["sentiment_norm_final"]
        + weights["speed"] * (1 - out["latency_norm"])
        + weights["brevity"] * (1 - out["length_norm"])
        + weights["reference"] * out["reference_similarity"].fillna(0.0)
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per (model, prompt), best hybrid_score first."""
    summary = (
        df.groupby(["model", "prompt_name"])
        .agg(
            human_rating=("human_rating", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            latency_seconds=("latency_seconds", "mean"),
            hybrid_score=("hybrid_score", "mean"),
            n=("hybrid_score", "size"),
            reference_similarity=("reference_similarity", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v1", help="scoring profile (default: v1, the original formula)")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    bench = pd.read_csv(args.bench)
    human = pd.read_csv(args.human)
    print(f"Benchmark results loaded: {bench.shape}")
    print(f"Human ratings loaded:     {human.shape}")

    try:
        merged, unmatched = merge_ratings(bench, human)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    print(f"Matched {len(merged)} rated responses.")
    if unmatched["benchmark_unmatched"] or unmatched["ratings_unmatched"]:
        print(
            f"WARNING: {unmatched['benchmark_unmatched']} benchmark rows have no rating and "
            f"{unmatched['ratings_unmatched']} ratings match no benchmark row; they are left out."
        )

    scored = add_scores(merged, args.profile)
    summary = summarize(scored)

    print(f"\n===== FINAL HYBRID SCORE RANKING (profile {args.profile}) =====")
    print(summary.round(4).to_string(index=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / "merged_full_evaluation.csv"
    ranking_path = args.out_dir / "final_prompt_ranking.csv"
    scored.to_csv(merged_path, index=False)
    summary.to_csv(ranking_path, index=False)
    print(f"\nSaved:\n - {merged_path}\n - {ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
