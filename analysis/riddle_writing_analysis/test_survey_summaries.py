"""Regression checks for aggregate-only survey reproducibility."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.constants import FIXED_TOPICS
from analysis.riddle_writing_analysis import survey_data, survey_summaries


class AggregateSurveyTests(unittest.TestCase):
    def test_missing_aggregate_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(survey_data, "DATA_DIR", Path(directory)):
                with self.assertRaises(FileNotFoundError):
                    survey_data.load_aggregates()

    def test_unapproved_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"respondent_id": ["not-an-aggregate"]}).to_csv(
                path / "survey_audit.csv", index=False
            )
            with patch.object(survey_data, "DATA_DIR", path):
                with self.assertRaisesRegex(ValueError, "Unexpected aggregate columns"):
                    survey_data.load_aggregates()

    def test_topic_comparison_uses_public_scores(self) -> None:
        difficulty = pd.DataFrame(
            {
                "topic": [*FIXED_TOPICS, "Wild Card"],
                "mean_difficulty": [*range(1, len(FIXED_TOPICS) + 1), 4],
                "responses": 60,
            }
        )
        human = pd.DataFrame(
            {
                "topic": list(FIXED_TOPICS),
                "community": "eligible",
                "is_scored": True,
                "is_correct": [False] * 4 + [True] * 4,
            }
        )
        with (
            patch.object(survey_summaries, "load_human_evaluation", return_value=human),
            patch.object(survey_summaries, "eligible_communities", return_value={"eligible"}),
        ):
            result = survey_summaries.topic_difficulty_human_accuracy(difficulty)
        self.assertEqual(list(result["topic"]), list(FIXED_TOPICS))
        self.assertEqual(result["human_responses"].sum(), len(FIXED_TOPICS))
        self.assertTrue(result["writing_responses"].eq(60).all())
        self.assertTrue(result["evaluated_communities"].eq(1).all())
        self.assertGreater(result["spearman_rho"].iloc[0], 0)

    def test_missing_fixed_topic_is_an_error(self) -> None:
        difficulty = pd.DataFrame(
            {
                "topic": list(FIXED_TOPICS[:-1]),
                "mean_difficulty": 3,
                "responses": 60,
            }
        )
        human = pd.DataFrame(
            {
                "topic": list(FIXED_TOPICS),
                "community": "eligible",
                "is_scored": True,
                "is_correct": True,
            }
        )
        with (
            patch.object(survey_summaries, "load_human_evaluation", return_value=human),
            patch.object(survey_summaries, "eligible_communities", return_value={"eligible"}),
        ):
            with self.assertRaisesRegex(ValueError, "Expected all fixed topics"):
                survey_summaries.topic_difficulty_human_accuracy(difficulty)


if __name__ == "__main__":
    unittest.main()
