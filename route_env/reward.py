from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from route_env.hf_client import extract_json
from route_env.io import read_jsonl
from route_env.verify import verify_prediction


@dataclass
class RewardResult:
    reward: float
    metrics: dict[str, Any]
    prediction: dict[str, Any]


class RouteRewarder:
    def __init__(self, tasks_path: str) -> None:
        self.tasks = {task["task_id"]: task for task in read_jsonl(tasks_path)}

    def score_response(self, task_id: str, response_text: str) -> RewardResult:
        task = self.tasks[task_id]
        try:
            prediction = extract_json(response_text)
            metrics = verify_prediction(task, {"task_id": task_id, "prediction": prediction})
        except Exception as exc:
            prediction = {"turns": None}
            metrics = {
                "task_id": task_id,
                "valid_schema": False,
                "valid_route": False,
                "score": 0.0,
                "error": str(exc),
            }
        return RewardResult(reward=float(metrics.get("score", 0.0)), metrics=metrics, prediction=prediction)


def reward_response(tasks_path: str, task_id: str, response_text: str) -> tuple[float, dict[str, Any]]:
    result = RouteRewarder(tasks_path).score_response(task_id, response_text)
    return result.reward, result.metrics
