from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from route_env.trace_choice_env import TraceChoiceEnv
from test_verify import make_linear_task


class TraceChoiceEnvTest(unittest.TestCase):
    def test_candidates_are_directed_from_current_frontier(self) -> None:
        task = make_linear_task(num_turns=3)
        env = TraceChoiceEnv(task, max_steps=8, trace_length_m=80)
        candidates = env.candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_path[:2], ["1", "2"])
        self.assertEqual(candidates[0].checkpoint_labels, ["T1"])

    def test_shortest_policy_reaches_destination_on_linear_task(self) -> None:
        task = make_linear_task(num_turns=4)
        env = TraceChoiceEnv(task, max_steps=16, trace_length_m=120)
        while not env.done:
            candidate_id = env.shortest_candidate_id()
            self.assertIsNotNone(candidate_id)
            env.step({"tool": "choose", "candidate_id": candidate_id})
        self.assertEqual(env.current_node, env.destination_node)
        self.assertTrue(env.last_metrics["valid_route"])
        self.assertAlmostEqual(env.last_metrics["length_ratio"], 1.0)

    def test_invalid_candidate_is_rejected(self) -> None:
        task = make_linear_task(num_turns=3)
        env = TraceChoiceEnv(task, max_steps=8)
        outcome = env.step({"tool": "choose", "candidate_id": "C99"})
        self.assertEqual(outcome.error, "unknown candidate_id: C99")
        self.assertEqual(env.route_nodes, ["1"])

    def test_destination_candidate_suppresses_detour_candidates(self) -> None:
        task = make_linear_task(num_turns=2)
        task["graph"]["edges"].append(
            {
                "u": "1",
                "v": "4",
                "length_m": 50.0,
                "geometry": [[0.0, 0.0], [0.003, 0.0]],
                "oneway": True,
                "highway": "residential",
            }
        )
        env = TraceChoiceEnv(task, max_steps=4, trace_length_m=80)
        candidates = env.candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].to_node, env.destination_node)
        self.assertTrue(candidates[0].reachable_to_destination)

    def test_unreachable_candidates_are_hidden_when_reachable_options_exist(self) -> None:
        task = make_linear_task(num_turns=2)
        task["graph"]["nodes"]["99"] = {"lat": 0.001, "lon": 0.0}
        task["graph"]["edges"].append(
            {
                "u": "1",
                "v": "99",
                "length_m": 25.0,
                "geometry": [[0.0, 0.0], [0.0, 0.001]],
                "oneway": True,
                "highway": "residential",
            }
        )
        env = TraceChoiceEnv(task, max_steps=4, trace_length_m=80)
        self.assertNotIn("99", [candidate.to_node for candidate in env.candidates()])

    def test_done_observation_has_no_candidates(self) -> None:
        task = make_linear_task(num_turns=1)
        env = TraceChoiceEnv(task, max_steps=8, trace_length_m=400)
        outcome = env.step({"tool": "choose", "candidate_id": "C1"})
        self.assertTrue(outcome.done)
        self.assertEqual(outcome.observation["candidates"], [])

    def test_exact_path_is_not_penalized_for_incidental_checkpoint_labels(self) -> None:
        task = make_linear_task(num_turns=3)
        task["oracle"]["gold_turn_route"] = ["T1", "T3"]
        env = TraceChoiceEnv(task, max_steps=8, trace_length_m=400)
        while not env.done:
            candidate_id = env.shortest_candidate_id()
            self.assertIsNotNone(candidate_id)
            env.step({"tool": "choose", "candidate_id": candidate_id})
        self.assertEqual(env.prediction()["turns"], ["T1", "T2", "T3"])
        self.assertEqual(env.scored_prediction()["turns"], ["T1", "T3"])
        self.assertTrue(env.last_metrics["valid_route"])
        self.assertAlmostEqual(env.last_metrics["score"], 1.0)

    def test_local_overview_render_writes_observation_image(self) -> None:
        task = make_linear_task(num_turns=2)
        with TemporaryDirectory() as tmp:
            env = TraceChoiceEnv(task, max_steps=4, trace_length_m=80, render_dir=tmp, render_context="local_overview")
            observation = env.observe()
            self.assertEqual(observation["view"]["context"], "local_overview")
            self.assertTrue(Path(observation["view"]["image"]).exists())


if __name__ == "__main__":
    unittest.main()
