from __future__ import annotations

import unittest

from route_env.map_env import CuaLiteMapEnv, oracle_actions_for_task
from test_verify import make_linear_task


def make_synthetic_strip() -> dict:
    first = make_linear_task(num_turns=2)
    first["task_id"] = "synthetic_strip_s01"
    first["segment_id"] = "S01"
    first["images"]["map"] = "data/test/s01.png"

    second = make_linear_task(num_turns=2)
    second["task_id"] = "synthetic_strip_s02"
    second["segment_id"] = "S02"
    second["images"]["map"] = "data/test/s02.png"

    return {
        "task_id": "synthetic_strip",
        "task_type": "route_strip",
        "images": {
            "overview": "data/test/overview.png",
            "segments": ["data/test/s01.png", "data/test/s02.png"],
        },
        "segments": [first, second],
        "oracle": {
            "distance_m": first["oracle"]["distance_m"] + second["oracle"]["distance_m"],
            "geometry": first["oracle"]["geometry"] + second["oracle"]["geometry"][1:],
        },
    }


class CuaLiteMapEnvTest(unittest.TestCase):
    def test_flat_oracle_trajectory_scores_perfectly(self) -> None:
        task = make_linear_task(num_turns=2)
        env = CuaLiteMapEnv(task, max_actions=16)
        for action in oracle_actions_for_task(task):
            outcome = env.step(action)
        self.assertTrue(outcome.done)
        self.assertEqual(env.prediction(), {"turns": ["T1", "T2"]})
        self.assertTrue(env.last_metrics["valid_route"])
        self.assertAlmostEqual(env.last_metrics["score"], 1.0)

    def test_route_strip_oracle_trajectory_scores_perfectly(self) -> None:
        task = make_synthetic_strip()
        env = CuaLiteMapEnv(task, max_actions=16)
        for action in oracle_actions_for_task(task):
            outcome = env.step(action)
        self.assertTrue(outcome.done)
        self.assertEqual(
            env.prediction(),
            {
                "segments": [
                    {"segment_id": "S01", "turns": ["T1", "T2"]},
                    {"segment_id": "S02", "turns": ["T1", "T2"]},
                ]
            },
        )
        self.assertTrue(env.last_metrics["valid_route"])
        self.assertAlmostEqual(env.last_metrics["score"], 1.0)

    def test_invalid_label_is_rejected_without_marking(self) -> None:
        task = make_synthetic_strip()
        env = CuaLiteMapEnv(task, max_actions=8)
        env.step({"tool": "open_segment", "segment_id": "S01"})
        outcome = env.step({"tool": "mark", "turn": "T99"})
        self.assertEqual(outcome.error, "T99 is not visible in S01")
        self.assertEqual(env.marked_by_segment["S01"], [])
        self.assertLess(outcome.step_reward, 0.0)

    def test_duplicate_mark_is_rejected(self) -> None:
        task = make_synthetic_strip()
        env = CuaLiteMapEnv(task, max_actions=8)
        env.step({"tool": "open_segment", "segment_id": "S01"})
        first = env.step({"tool": "mark", "turn": "T1"})
        second = env.step({"tool": "mark", "turn": "T1"})
        self.assertIsNone(first.error)
        self.assertEqual(second.error, "T1 is already marked in S01")
        self.assertEqual(env.marked_by_segment["S01"], ["T1"])
        self.assertLess(second.step_reward, 0.0)

    def test_zoom_to_label_filters_visible_labels(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8)
        before = env.observe()["visible_labels"]
        outcome = env.step({"tool": "zoom_to_label", "turn": "T1"})
        after = outcome.observation["visible_labels"]
        self.assertIn("T1", after)
        self.assertLess(len(after), len(before))

    def test_start_view_is_local(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8, initial_view="start", viewport_scale=0.3)
        observation = env.observe()
        self.assertIn("T1", observation["visible_labels"])
        self.assertNotIn("T8", observation["visible_labels"])
        self.assertLess(observation["view"]["span"]["lon"], task["task_bbox"][2] - task["task_bbox"][0])

    def test_flat_mark_rejects_label_outside_current_viewport(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8, initial_view="start", viewport_scale=0.3)
        outcome = env.step({"tool": "mark", "turn": "T8"})
        self.assertEqual(outcome.error, "T8 is not visible in the current viewport")
        self.assertEqual(env.prediction(), {"turns": []})

    def test_center_and_pan_toward_destination_moves_local_view(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8, initial_view="start", viewport_scale=0.3)
        start_center = env.observe()["view"]["center"]["lon"]
        env.step({"tool": "mark", "turn": "T2"})
        env.step({"tool": "center_on_last_mark"})
        centered = env.observe()["view"]["center"]["lon"]
        env.step({"tool": "pan_toward_destination"})
        panned = env.observe()["view"]["center"]["lon"]
        self.assertGreater(centered, start_center)
        self.assertGreater(panned, centered)

    def test_visible_label_throttle_limits_prompt_and_render_labels(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8, max_visible_labels=3)
        observation = env.observe()
        self.assertEqual(len(observation["visible_labels"]), 3)

    def test_frontier_candidates_are_ordered_from_current_frontier(self) -> None:
        task = make_linear_task(num_turns=8)
        env = CuaLiteMapEnv(task, max_actions=8)
        start = env.observe()
        self.assertEqual(start["frontier_candidates"][:3], ["T1", "T2", "T3"])
        env.step({"tool": "mark", "turn": "T4"})
        after_mark = env.observe()
        self.assertEqual(after_mark["frontier_candidates"][:2], ["T3", "T5"])

    def test_prefix_validity_mask_removes_unroutable_backtracking_labels(self) -> None:
        task = make_linear_task(num_turns=4)
        env = CuaLiteMapEnv(task, max_actions=8, enforce_prefix_validity=True)
        env.step({"tool": "mark", "turn": "T2"})
        observation = env.observe()
        self.assertNotIn("T1", observation["markable_labels"])
        outcome = env.step({"tool": "mark", "turn": "T1"})
        self.assertEqual(outcome.error, "T1 would make the route prefix unroutable")
        self.assertEqual(env.prediction(), {"turns": ["T2"]})

    def test_observation_exposes_current_panel_and_visible_labels(self) -> None:
        task = make_synthetic_strip()
        env = CuaLiteMapEnv(task, max_actions=8)
        overview = env.observe()
        self.assertEqual(overview["view"]["kind"], "overview")
        self.assertEqual(overview["visible_segments"], ["S01", "S02"])
        self.assertEqual(overview["visible_labels"], [])

        segment = env.step({"tool": "open_segment", "segment_id": "S02"}).observation
        self.assertEqual(segment["view"]["kind"], "segment")
        self.assertEqual(segment["view"]["segment_id"], "S02")
        self.assertEqual(segment["visible_labels"], ["T1", "T2"])


if __name__ == "__main__":
    unittest.main()
