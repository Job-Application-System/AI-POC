import json
import unittest
from unittest.mock import patch

from document_layout import HEADER_LINKS, conciseness_prompt, layout_rewards
import cover_letter_reward as reward
import resume_reward


class LayoutTests(unittest.TestCase):
    def test_page_limits_include_header(self):
        header = "Kelvin Mock | Ottawa | kelvin.mock@gmail.com\n" + "\n".join(HEADER_LINKS)
        one_page = header + "\n\nSUMMARY\nBuilt reliable software."
        two_pages = one_page + "\fExperience"
        three_pages = two_pages + "\fEducation"
        self.assertEqual(layout_rewards(one_page, "cover_letter")["header_reward"], 1)
        self.assertEqual(layout_rewards(one_page, "cover_letter")["page_fit_reward"], 1)
        self.assertEqual(layout_rewards(two_pages, "cover_letter")["page_fit_reward"], 0)
        self.assertEqual(layout_rewards(two_pages, "resume")["page_fit_reward"], 1)
        self.assertEqual(layout_rewards(three_pages, "resume")["page_fit_reward"], 0)
        self.assertEqual(layout_rewards("", "resume")["page_fit_reward"], 0)

    def test_natural_overflow(self):
        self.assertGreater(layout_rewards("Experience\n" + "Implemented reliable services.\n" * 140, "resume")["page_count"], 2)

    def test_shared_scorer_and_resume_entrypoint(self):
        with patch.object(reward, "_similarity_scores", return_value={}), patch.object(reward, "_length_reward", return_value=0.7):
            text = "Kelvin Mock\fExperience"
            cover = reward.compute_score(text, json.dumps({}))
            resume = resume_reward.compute_score(text, json.dumps({}))
        self.assertEqual(cover["length_reward"], 0.7)
        self.assertAlmostEqual(resume["score"] - cover["score"], 0.1)

    def test_prompts(self):
        for task, limit in [("resume", 2), ("cover_letter", 1)]:
            prompt = conciseness_prompt(task)
            self.assertIn(f"{limit} page(s)", prompt)
            self.assertIn("Times New Roman", prompt)
            self.assertNotIn("Letter paper", prompt)
            self.assertTrue(all(link in prompt for link in HEADER_LINKS))


if __name__ == "__main__":
    unittest.main()
