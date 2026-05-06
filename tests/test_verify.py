from __future__ import annotations

import unittest

from route_env.verify import verify_prediction


def make_linear_task(num_turns: int = 2) -> dict:
    node_count = num_turns + 2
    nodes = {
        str(i + 1): {"lat": 0.0, "lon": i * 0.001}
        for i in range(node_count)
    }
    edges = []
    oracle_geometry = []
    for i in range(node_count):
        oracle_geometry.append([i * 0.001, 0.0])
        if i < node_count - 1:
            edges.append(
                {
                    "u": str(i + 1),
                    "v": str(i + 2),
                    "length_m": 100.0,
                    "geometry": [[i * 0.001, 0.0], [(i + 1) * 0.001, 0.0]],
                    "oneway": True,
                    "highway": "residential",
                }
            )
    checkpoints = {
        f"T{i}": {"lat": 0.0, "lon": i * 0.001, "osm_id": i + 1}
        for i in range(1, num_turns + 1)
    }
    return {
        "task_id": "synthetic_linear",
        "city": "Synthetic",
        "mode": "drive",
        "network_type": "drive",
        "bbox": [0.0, 0.0, node_count * 0.001, 0.001],
        "task_bbox": [0.0, 0.0, node_count * 0.001, 0.001],
        "origin": {"lat": 0.0, "lon": 0.0, "osm_id": 1, "label": "A"},
        "destination": {"lat": 0.0, "lon": (node_count - 1) * 0.001, "osm_id": node_count, "label": "B"},
        "images": {"map": "data/rendered/synthetic_linear/map.png"},
        "turn_checkpoints": checkpoints,
        "graph": {"nodes": nodes, "edges": edges},
        "oracle": {
            "provider": "synthetic",
            "distance_m": 100.0 * (node_count - 1),
            "geometry": oracle_geometry,
            "gold_turn_route": list(checkpoints),
            "gold_osm_route": list(range(1, node_count + 1)),
            "turn_count": num_turns,
        },
    }


class VerifyPredictionTest(unittest.TestCase):
    def test_oracle_scores_perfectly(self) -> None:
        task = make_linear_task()
        result = verify_prediction(task, {"prediction": {"turns": ["T1", "T2"], "confidence": 1.0}})
        self.assertTrue(result["valid_schema"])
        self.assertTrue(result["valid_route"])
        self.assertEqual(result["checkpoint_reward"], 1.0)
        self.assertAlmostEqual(result["score"], 1.0)

    def test_empty_turns_cannot_get_success_score(self) -> None:
        task = make_linear_task()
        result = verify_prediction(task, {"prediction": {"turns": [], "confidence": 0.0}})
        self.assertFalse(result["valid_route"])
        self.assertEqual(result["checkpoint_reward"], 0.0)
        self.assertLess(result["score"], 0.75)

    def test_unknown_turn_invalidates_route(self) -> None:
        task = make_linear_task()
        result = verify_prediction(task, {"prediction": {"turns": ["T1", "T99"], "confidence": 0.5}})
        self.assertFalse(result["valid_route"])
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["unknown_turns"], ["T99"])

    def test_non_string_turns_fail_schema_and_route(self) -> None:
        task = make_linear_task()
        result = verify_prediction(task, {"prediction": {"turns": ["T1", 2], "confidence": 0.5}})
        self.assertFalse(result["valid_schema"])
        self.assertFalse(result["valid_turn_list"])
        self.assertFalse(result["valid_route"])

    def test_long_gold_route_is_not_penalized_for_its_own_turn_count(self) -> None:
        task = make_linear_task(num_turns=16)
        gold = task["oracle"]["gold_turn_route"]
        result = verify_prediction(task, {"prediction": {"turns": gold, "confidence": 1.0}})
        self.assertTrue(result["valid_route"])
        self.assertEqual(result["turn_count_penalty"], 1.0)
        self.assertAlmostEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
