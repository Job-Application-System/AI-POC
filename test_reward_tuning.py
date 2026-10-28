import json
import random
import tempfile
import unittest
from pathlib import Path

import training_pipeline as pipeline
from cover_letter_reward import _weighted_similarity


class RewardTuningTests(unittest.TestCase):
    def test_all_weights_in_search(self):
        params = pipeline.suggest_tuning_params(pipeline.RandomSuggestionTrial(random.Random(3)))
        self.assertEqual(set(params), set(pipeline.DEFAULT_TUNING_PARAMS))
        self.assertEqual(len(params), 17)  # 13 reward weights + 4 generation settings
        self.assertAlmostEqual(sum(pipeline.tuning_blend_weights(params).values()), 1)
        scores = {"final_score": 1}
        params["final_score_weight"] = 0
        self.assertEqual(pipeline.weighted_similarity(scores, params), 0)
        self.assertEqual(_weighted_similarity(scores, params, 0), 0)

    def test_cv_selection_persistence_and_rows(self):
        default = dict(pipeline.DEFAULT_TUNING_PARAMS)
        page_only = {**default, **{f"blend_{key}": float(key == "page_fit_reward")
                                  for key in pipeline.DEFAULT_REWARD_BLEND_WEIGHTS}}
        def candidate(fits):
            return {
                "cover_letter": "Fits" if fits else "Overflow",
                "component_scores": {"final_score": 0.8 if fits else 1.0},
                "reward_metrics": {"length_reward": 1, "format_reward": 1,
                                   "candidate_match": 0, "page_fit_reward": float(fits),
                                   "header_reward": 1},
            }
        candidates = [candidate(False), candidate(True)]
        self.assertEqual(pipeline.best_candidate(candidates, page_only, "cover_letter")["cover_letter"], "Fits")
        with tempfile.TemporaryDirectory() as directory:
            selected = pipeline.tune_reward_weights(
                {"cover_letter": {0: candidates, 1: candidates}}, 2, 3,
                Path(directory), 1, [default, page_only])
            saved = json.loads((Path(directory) / "cv_reward_weights.json").read_text())
        self.assertEqual(len(selected), 17)
        self.assertIn("selected_reward_blend_weights", saved)
        example = pipeline.JobExample("job", "resume", "profile", "company")
        for make_row in [pipeline.to_cover_letter_row, pipeline.to_resume_row]:
            truth = json.loads(make_row(example, 0, reward_weights=page_only)["reward_model"]["ground_truth"])
            self.assertEqual(truth["reward_blend_weights"]["page_fit_reward"], 1)
            self.assertEqual(truth["final_score_weight"], page_only["final_score_weight"])


if __name__ == "__main__":
    unittest.main()
