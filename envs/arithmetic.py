import re

from .base import Env


class ArithmeticEnv(Env):
    """Tiny rule-reward environment for arithmetic smoke tests."""

    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False

    def _extract_answer(self, response_str):
        matches = re.findall(r"<answer>(.*?)</answer>", response_str, flags=re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        scores = []
        format_bonus = 0.0 if if_val else 0.1

        for i in range(len(data)):
            processed_data = self._process_data(data_item=data[i], tokenizer=tokenizer)
            ground_truth = processed_data["ground_truth"]
            response_str = processed_data["response_str"]

            prediction = self._extract_answer(response_str)
            target = str(ground_truth["target"]).strip()

            if prediction is None:
                score = -format_bonus
            else:
                normalized_prediction = prediction.replace(",", "").strip()
                score = 1.0 if normalized_prediction == target else 0.0
                score += format_bonus

            scores.append([score])

        return scores
