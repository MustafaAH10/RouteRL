# Graph-First RouteRL Direction

This is a proposed pivot away from pure VLM free-form checkpoint prediction.

The core idea: use OSM-derived directed road graphs as the primary environment,
teach a generic model to approximate efficient valid routes from graph features,
and use VLM/map rendering later as an auxiliary perception and reranking layer.

## Why Pivot

The previous CUA-lite VLM interface:

```text
inspect(target)
edit_route(turns)
preview_route(turns)
finish(turns)
```

was useful for debugging, but it is too unconstrained and too visual for the
main problem. The model can still hallucinate sparse checkpoint labels, miss
directed-road constraints, or overfit to cluttered labels.

For a Google Maps alternative, the real asset is the graph:

```text
directed roads
edge distances
road class
turn/checkpoint nodes
start and destination
```

A model should first learn to route in that graph. Images can come later.

## Important Reality Check

Dijkstra is not expensive on the current per-task graphs:

```text
easy:   mean 84 nodes,    max 264
medium: mean 429 nodes,   max 1289
hard:   mean 3519 nodes,  max 12014
```

So the goal should not be “avoid Dijkstra because Dijkstra is impossible.”

The better framing is:

```text
Dijkstra / shortest path = teacher and verifier
model = fast prior, candidate generator, policy, reranker, RL learner
```

The model can learn route optimization behavior, propose likely corridors, and
serve cases where you want route reasoning rather than a black-box API call.

## Working Versions Implemented

### Version A: Local Greedy Numeric Router

Script:

```text
scripts/run_graph_route_approx_policies.py
```

Policy:

```text
At each node, choose the outgoing directed edge that looks best by:
- edge length
- straight-line distance to destination
- progress toward destination
- revisit penalty
```

No Dijkstra is used to choose the route.

Result: this fails on longer graphs. It is too myopic and gets trapped in loops
or locally attractive wrong roads.

### Version B: Hill-Climb Beam Router

Script:

```text
scripts/run_graph_route_approx_policies.py
```

Policy:

```text
At each step, look ahead a few local moves with a small beam.
Commit one next edge.
Repeat.
```

No Dijkstra is used to choose the route.

Result: better than pure greedy on short tasks, but still collapses on hard
routes. It makes irreversible local mistakes.

### Version C: Heuristic Beam Router

Script:

```text
scripts/run_graph_route_approx_policies.py
```

Policy:

```text
Maintain a beam of partial paths.
Rank partial paths by:
  path length so far
  straight-line distance to destination
  loop penalty
Stop when a destination path is found.
```

This is close to an A*-style learned-policy scaffold, but it does not use
Dijkstra or hidden shortest-path distances to choose the route.

This is the most promising non-neural baseline so far.

### Version D: Graph-Text LLM Task Format

Script:

```text
scripts/make_graph_route_text_tasks.py
```

This exports a task as:

```json
{
  "start_node": "...",
  "goal_node": "...",
  "graph": {
    "nodes": [
      {"id":"...", "lat":1.23, "lon":103.8, "straight_to_goal_m":1234.5}
    ],
    "edges": [
      {"u":"...", "v":"...", "length_m":42.0, "bearing_deg":91.2, "highway":"primary"}
    ]
  },
  "target": {
    "route_nodes": ["..."],
    "turns": ["T1", "T2"],
    "distance_m": 1234.5
  }
}
```

This is the cleanest path to training a generic LLM on graph routing first.
The target route is produced by the hidden shortest-path teacher and is used
only for training/evaluation.

Sample files:

```text
data/experiments/singapore_benchmark_ladder_1500/easy/graph_text_route_tasks_sample10.jsonl
data/experiments/singapore_benchmark_ladder_1500/hard/graph_text_route_tasks_sample3.jsonl
```

### Version E: Graph Policy / Learned Heuristic States

Script:

```text
scripts/make_graph_policy_dataset.py
```

This is the better LLM/RL novelty target.

Instead of asking an LLM to pick from already-ranked full routes, each training
record is one graph decision:

```json
{
  "current_node": "2668132709",
  "goal_node": "13192292919",
  "route_so_far_m": 0,
  "remaining_teacher_steps": 9,
  "candidates": [
    {
      "candidate_id": "C1",
      "to_node": "245396695",
      "edge_length_m": 16.6,
      "progress_m": 3.0,
      "straight_to_goal_m": 604.9,
      "bearing_deg": 172.0,
      "highway": ""
    }
  ],
  "target": {
    "candidate_id": "C1",
    "next_node": "245396695"
  }
}
```

This can train:

```text
LLM/SFT policy:     choose the teacher next edge
RL policy:          choose legal edge actions and optimize final route reward
learned heuristic:  score outgoing edges/frontier states to guide beam search
```

Sample files:

```text
data/experiments/singapore_benchmark_ladder_1500/easy/graph_policy_states_sample20.jsonl
data/experiments/singapore_benchmark_ladder_1500/hard/graph_policy_states_sample5.jsonl
```

Full Singapore benchmark exports:

```text
data/experiments/singapore_benchmark_ladder_1500/easy/graph_policy_states_all500.jsonl
data/experiments/singapore_benchmark_ladder_1500/medium/graph_policy_states_all500.jsonl
data/experiments/singapore_benchmark_ladder_1500/hard/graph_policy_states_all500.jsonl
```

This is more novel than a route-candidate ranker because the model is part of
the search process rather than a cosmetic selector after the graph code already
found a near-best route.

### Version F: Graph Policy State Baseline

Script:

```text
scripts/evaluate_graph_policy_states.py
```

This evaluates simple next-edge policies over the graph-policy states. The key
metric is branching accuracy, not raw accuracy, because a large minority of
states have only one legal outgoing road edge.

Policies:

```text
random
shortest_edge
closest_to_goal
max_progress
progress_per_meter
nonvisited_closest_goal
oracle
```

The best dumb baseline is currently `progress_per_meter`: choose the outgoing
edge with the most straight-line progress per meter driven. It is strong, but
not perfect. That gap is the first clean target for a learned graph policy.

## Benchmark Results

### Easy, 500 Tasks

```text
heuristic_beam mean_score:        0.9597
heuristic_beam valid_route_rate:  0.998
heuristic_beam success@0.75:      0.940

hill_climb_beam mean_score:       0.6464
local_greedy mean_score:          0.5872
```

### Medium, 500 Tasks

```text
heuristic_beam mean_score:        0.8594
heuristic_beam valid_route_rate:  0.946
heuristic_beam success@0.75:      0.800

hill_climb_beam mean_score:       0.3748
local_greedy mean_score:          0.2537
```

### Hard, 8-25 km, 500 Tasks

Heuristic beam only, full 500:

```text
heuristic_beam mean_score:        0.6705
heuristic_beam valid_route_rate:  0.818
heuristic_beam success@0.75:      0.500
heuristic_beam mean_length_ratio: 1.0637
```

Hard 100-task comparison:

```text
heuristic_beam mean_score:        0.6783
hill_climb_beam mean_score:       0.1514
local_greedy mean_score:          0.0991
```

The result is pretty clear: local greedy and small local lookahead are not
enough. A model needs either broader search, learned global priors, or explicit
candidate route generation.

### Graph Policy Next-Edge Baselines

Full graph-policy export:

```text
easy:   7,204 states from 500 tasks
medium: 18,938 states from 500 tasks
hard:   51,105 states from 500 tasks
total:  77,247 states from 1,500 tasks
```

Branching states are the meaningful states where the model has more than one
legal outgoing edge:

```text
easy branching states:   5,358
medium branching states: 14,125
hard branching states:   36,748
total branching states:  56,231
```

Branching accuracy:

| Split | Random | Shortest edge | Closest to goal | Progress/meter | Nonvisited closest | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| easy | `0.448` | `0.469` | `0.842` | `0.857` | `0.863` | `1.000` |
| medium | `0.468` | `0.480` | `0.814` | `0.856` | `0.824` | `1.000` |
| hard | `0.482` | `0.484` | `0.802` | `0.852` | `0.808` | `1.000` |
| cumulative | `0.475` | `0.482` | `0.809` | `0.854` | `0.817` | `1.000` |

Files:

```text
data/experiments/singapore_benchmark_ladder_1500/summary_06_graph_policy_state_baselines.json
data/experiments/singapore_benchmark_ladder_1500/easy/results/06_graph_policy_state_baselines_all500.summary.json
data/experiments/singapore_benchmark_ladder_1500/medium/results/06_graph_policy_state_baselines_all500.summary.json
data/experiments/singapore_benchmark_ladder_1500/hard/results/06_graph_policy_state_baselines_all500.summary.json
```

Interpretation:

```text
The shortest-edge baseline is weak.
Straight-line geometric progress is strong.
Local next-edge accuracy still has a real 14-19 point gap to oracle on branching states.
Even high next-edge accuracy can compound into bad full-route rollouts on hard tasks.
```

This is a better training target than route-candidate reranking. An LLM/RL
policy can learn when the obvious geometric move is wrong, then act as a learned
frontier scorer inside beam search.

## Best Path Forward

I would reframe RouteRL as:

```text
Graph policy first.
Visual grounding second.
Browser/map GUI demo third.
```

Recommended ladder:

### 1. Graph Oracle Teacher

Use Dijkstra/shortest path to create targets.

This is not the model. This is the teacher/verifier.

### 2. Graph Approximation Baselines

Keep:

```text
local_greedy
hill_climb_beam
heuristic_beam
```

These provide non-neural baselines.

### 3. LLM Graph-Text One-Shot

Train or prompt a generic model on compact adjacency-list records:

```text
input: directed graph + start + goal
output: route_nodes
```

Start on easy and medium. Do not start with 8-25 km.

### 4. Candidate-Route Reranking

Generate 5-20 valid candidate routes using cheap heuristics and perturbations.

Ask an LLM/VLM to choose/rank:

```text
shortest, least loopy, most plausible route
```

This is useful for product demos and preference routing, but it is not the main
novelty if candidate distances are already visible. If all candidates are valid
and the shortest distance is exposed, the best baseline is just "choose shortest."
The LLM only adds value here when:

```text
the route objective is richer than distance
the candidate generator is weak and needs learned judgment
visual/user-preference constraints matter
```

### 5. Learned Heuristic / Graph Policy

Use reward:

```text
valid destination reached
length ratio
loop penalty
geometry similarity
checkpoint alignment
```

Train a policy on graph decision states to improve over straight-line heuristic
beam. The model should score legal next edges or frontier states; the environment
keeps validity structural.

This is the strongest LLM/RL novelty:

```text
Dijkstra teacher creates supervision.
LLM/policy learns edge/frontier decisions.
RL fine-tunes for route length, validity, loops, preferences.
Beam search uses learned scores instead of only geometric distance.
```

The first model benchmark should not be a route ranker. It should be:

```text
input:  current node, goal, route-so-far, legal outgoing edges
output: one candidate_id
```

Then use that model score inside a rollout:

```text
current frontier -> model scores candidate edges -> beam expands -> verifier scores final route
```

That gives clear novelty:

```text
SFT learns from shortest-path teacher actions.
RL improves the learned scorer against final-route reward.
Inference uses the model as a graph-search prior, not as a cosmetic selector.
```

### 6. VLM Layer

Use the VLM only where vision helps:

```text
inspect rendered route candidates
spot visually ugly detours
read map constraints if graph metadata is sparse
explain route choices to a user
```

Do not ask the VLM to freely invent a whole route from a cluttered map image.

## Product Implication

The viable low-cost alternative is probably not:

```text
VLM looks at a map screenshot and writes a route.
```

It is:

```text
OSM graph database
cheap graph candidate generator
LLM/RL policy or reranker
optional VLM route visualizer/explainer
```

This can still be visually appealing. The GUI can show the route, alternatives,
confidence, and reasoning. But the route engine should be graph-native.

## Concrete Next Experiments

1. Train a small text model or lightweight classifier on `graph_policy_states_all500.jsonl`.
2. Evaluate next-edge accuracy against `progress_per_meter`, especially on branching states.
3. Plug learned edge scores into beam search and measure full-route score on easy/medium/hard.
4. Fine-tune with RL on final route reward: valid destination, length ratio, loop penalty, and checkpoint alignment.
5. Add VLM only after the graph policy beats the geometric baseline in rollout, not just on static images.

This is a much cleaner road to RL than the previous CUA-lite tool setup.
