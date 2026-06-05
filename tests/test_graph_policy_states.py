from __future__ import annotations

import random
import unittest

from scripts.evaluate_graph_policy_states import evaluate_record, summarize


def make_state() -> dict:
    return {
        "task_id": "synthetic_route",
        "state_id": "synthetic_route::step_0000",
        "split": "test",
        "current_node": "A",
        "goal_node": "D",
        "candidates": [
            {
                "candidate_id": "C1",
                "to_node": "B",
                "edge_length_m": 10.0,
                "progress_m": 1.0,
                "straight_to_goal_m": 90.0,
                "visited": False,
            },
            {
                "candidate_id": "C2",
                "to_node": "C",
                "edge_length_m": 30.0,
                "progress_m": 40.0,
                "straight_to_goal_m": 51.0,
                "visited": False,
            },
        ],
        "target": {"candidate_id": "C2", "next_node": "C"},
    }


class GraphPolicyStateEvaluationTest(unittest.TestCase):
    def test_progress_policy_selects_teacher_candidate(self) -> None:
        record = evaluate_record(make_state(), "progress_per_meter", random.Random(0))

        self.assertEqual(record["selected_candidate_id"], "C2")
        self.assertEqual(record["target_candidate_id"], "C2")
        self.assertTrue(record["correct"])
        self.assertTrue(record["branching"])

    def test_shortest_edge_can_be_wrong_on_branching_state(self) -> None:
        record = evaluate_record(make_state(), "shortest_edge", random.Random(0))

        self.assertEqual(record["selected_candidate_id"], "C1")
        self.assertFalse(record["correct"])

    def test_summary_reports_branching_accuracy(self) -> None:
        records = [
            evaluate_record(make_state(), "progress_per_meter", random.Random(0)),
            evaluate_record(make_state(), "shortest_edge", random.Random(0)),
        ]

        summary = summarize(records)

        self.assertEqual(summary["progress_per_meter"]["branching_count"], 1)
        self.assertEqual(summary["progress_per_meter"]["branching_accuracy"], 1.0)
        self.assertEqual(summary["shortest_edge"]["branching_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
