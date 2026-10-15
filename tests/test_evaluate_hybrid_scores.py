"""Tests for scripts/evaluate/evaluate_hybrid_scores.py.

Run from the repo root (standard library only, no pytest needed):
    python -m unittest discover -s tests -v
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "evaluate" / "evaluate_hybrid_scores.py"

spec = importlib.util.spec_from_file_location("evaluate_hybrid_scores", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def bench_row(query, response, model="m1", prompt="p1", latency=1.0, length=10, sentiment=0.5, expected="ref"):
    return {
        "query_id": hash((query, response)) % 10_000,
        "customer_query": query,
        "model_response": response,
        "model": model,
        "prompt_name": prompt,
        "latency_seconds": latency,
        "response_length": length,
        "sentiment_score": sentiment,
        "expected_response": expected,
    }


class MinMaxTests(unittest.TestCase):
    def test_scales_to_unit_range(self):
        out = ev.minmax(pd.Series([2.0, 3.5, 5.0]))
        self.assertEqual(list(out), [0.0, 0.5, 1.0])

    def test_constant_column_is_zero_like_sklearn(self):
        self.assertEqual(list(ev.minmax(pd.Series([4.0, 4.0, 4.0]))), [0.0, 0.0, 0.0])

    def test_nan_passes_through(self):
        out = ev.minmax(pd.Series([1.0, np.nan, 3.0]))
        self.assertEqual(out.iloc[0], 0.0)
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertEqual(out.iloc[2], 1.0)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.bench = pd.DataFrame([bench_row("q1", "a"), bench_row("q2", "b"), bench_row("q3", "c")])

    def human(self, rows):
        return pd.DataFrame(rows, columns=["customer_query", "model_response", "human_rating"])

    def test_matches_on_query_and_response(self):
        merged, unmatched = ev.merge_ratings(self.bench, self.human([("q1", "a", 4), ("q2", "b", 2), ("q3", "c", 5)]))
        self.assertEqual(len(merged), 3)
        self.assertEqual(unmatched, {"benchmark_unmatched": 0, "ratings_unmatched": 0})
        self.assertEqual(dict(zip(merged.customer_query, merged.human_rating)), {"q1": 4, "q2": 2, "q3": 5})

    def test_identical_responses_to_different_queries_do_not_cross_match(self):
        # The previous merge keyed on model_response alone and would have multiplied these rows.
        bench = pd.DataFrame([bench_row("q1", "Sure, happy to help!"), bench_row("q2", "Sure, happy to help!")])
        merged, _ = ev.merge_ratings(bench, self.human([("q1", "Sure, happy to help!", 5), ("q2", "Sure, happy to help!", 2)]))
        self.assertEqual(len(merged), 2)
        self.assertEqual(dict(zip(merged.customer_query, merged.human_rating)), {"q1": 5, "q2": 2})

    def test_duplicate_keys_are_an_error_not_a_silent_row_multiplication(self):
        dup = self.human([("q1", "a", 4), ("q1", "a", 3), ("q2", "b", 2)])
        with self.assertRaisesRegex(ValueError, "human ratings share the same"):
            ev.merge_ratings(self.bench, dup)
        bench_dup = pd.concat([self.bench, self.bench.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "benchmark results share the same"):
            ev.merge_ratings(bench_dup, self.human([("q1", "a", 4)]))

    def test_partial_ratings_are_reported(self):
        merged, unmatched = ev.merge_ratings(self.bench, self.human([("q1", "a", 4), ("q9", "zzz", 3)]))
        self.assertEqual(len(merged), 1)
        self.assertEqual(unmatched, {"benchmark_unmatched": 2, "ratings_unmatched": 1})

    def test_unrated_and_failed_rows_are_ignored(self):
        bench = pd.concat([self.bench, pd.DataFrame([bench_row("q4", None)])], ignore_index=True)  # API call failed
        human = self.human([("q1", "a", 4), ("q2", "b", np.nan)])  # q2 not rated yet
        merged, _ = ev.merge_ratings(bench, human)
        self.assertEqual(list(merged.customer_query), ["q1"])

    def test_nothing_matching_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no benchmark row matched"):
            ev.merge_ratings(self.bench, self.human([("other", "text", 3)]))

    def test_missing_columns_are_named(self):
        with self.assertRaisesRegex(ValueError, "'human_rating'"):
            ev.merge_ratings(self.bench, pd.DataFrame({"customer_query": ["q1"], "model_response": ["a"]}))


class ReferenceSimilarityTests(unittest.TestCase):
    def test_identical_text_is_one_and_unrelated_is_zero(self):
        sims = ev.reference_similarity(
            pd.Series(["refund your order within thirty days", "refund your order within thirty days"]),
            pd.Series(["refund your order within thirty days", "the weather is sunny today"]),
        )
        self.assertAlmostEqual(sims.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(sims.iloc[1], 0.0, places=6)

    def test_missing_reference_is_nan_not_zero(self):
        sims = ev.reference_similarity(pd.Series(["a real answer here", "another real answer"]), pd.Series([None, "   "]))
        self.assertTrue(sims.isna().all())

    def test_texts_made_only_of_stop_words_give_nan_instead_of_crashing(self):
        sims = ev.reference_similarity(pd.Series(["it is the"]), pd.Series(["to be or not"]))
        self.assertTrue(sims.isna().all())

    def test_stays_in_unit_range(self):
        sims = ev.reference_similarity(
            pd.Series(["track my order please", "cancel the subscription now"]),
            pd.Series(["where is my order", "how do I cancel my subscription"]),
        )
        self.assertTrue(((sims >= 0) & (sims <= 1)).all())


class ScoringTests(unittest.TestCase):
    def frame(self):
        rows = [
            bench_row("q1", "refund within thirty days", latency=1.0, length=10, sentiment=0.9, expected="refund within thirty days"),
            bench_row("q2", "two words", latency=3.0, length=30, sentiment=-0.5, expected="unrelated text"),
        ]
        df = pd.DataFrame(rows)
        df["human_rating"] = [5, 2]
        return df

    def test_profile_weights_sum_to_one(self):
        for name, weights in ev.PROFILES.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=9, msg=name)

    def test_v1_scales_human_rating_over_the_observed_range(self):
        out = ev.add_scores(self.frame(), "v1")
        self.assertEqual(list(out.human_norm), [1.0, 0.0])  # a rating of 2 maps to 0.0 under v1

    def test_v2_uses_the_fixed_one_to_five_scale(self):
        out = ev.add_scores(self.frame(), "v2")
        self.assertEqual(list(out.human_norm), [1.0, 0.25])

    def test_v1_formula(self):
        out = ev.add_scores(self.frame(), "v1")
        best = 0.55 * 1 + 0.20 * 1 + 0.20 * 1 + 0.05 * 1  # top rating, top sentiment, fastest, shortest
        self.assertAlmostEqual(out.hybrid_score.iloc[0], best, places=9)
        self.assertAlmostEqual(out.hybrid_score.iloc[1], 0.0, places=9)

    def test_v2_ignores_sentiment_and_rewards_reference_similarity(self):
        df = self.frame()
        base = ev.add_scores(df, "v2").hybrid_score
        flipped = df.copy()
        flipped["sentiment_score"] = flipped["sentiment_score"].iloc[::-1].values
        self.assertTrue(np.allclose(base, ev.add_scores(flipped, "v2").hybrid_score))
        self.assertGreater(ev.add_scores(df, "v2").reference_similarity.iloc[0], 0.9)

    def test_scores_are_between_zero_and_one(self):
        for profile in ev.PROFILES:
            out = ev.add_scores(self.frame(), profile)
            self.assertTrue(((out.hybrid_score >= -1e-9) & (out.hybrid_score <= 1 + 1e-9)).all(), profile)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown profile"):
            ev.add_scores(self.frame(), "v9")


class CommittedResultsRegressionTests(unittest.TestCase):
    """v1 must keep reproducing the published ranking, otherwise earlier conclusions silently change."""

    def test_v1_reproduces_final_prompt_ranking_csv(self):
        committed = pd.read_csv(REPO_ROOT / "results" / "evaluation" / "final_prompt_ranking.csv")
        with tempfile.TemporaryDirectory() as out_dir:
            self.assertEqual(ev.main(["--profile", "v1", "--out-dir", out_dir]), 0)
            fresh = pd.read_csv(Path(out_dir) / "final_prompt_ranking.csv")

        # Original columns first, in the original order; new columns only appended.
        self.assertEqual(list(fresh.columns[: len(committed.columns)]), list(committed.columns))
        key = ["model", "prompt_name"]
        a = committed.set_index(key).sort_index()
        b = fresh.set_index(key)[list(a.columns)].sort_index()
        self.assertTrue((a.index == b.index).all())
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-9, rtol=0)
        # ...and the winner is still gpt-4o with the friendly prompt.
        self.assertEqual(tuple(fresh.iloc[0][key]), ("gpt-4o", "expanded_friendly"))

    def test_main_reports_a_mismatched_pair_of_files_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            human = Path(tmp) / "human.csv"
            pd.DataFrame({"customer_query": ["x"], "model_response": ["y"], "human_rating": [3]}).to_csv(human, index=False)
            self.assertEqual(ev.main(["--human", str(human), "--out-dir", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
