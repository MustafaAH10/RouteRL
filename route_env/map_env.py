from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from route_env.io import read_jsonl
from route_env.render import render_task
from route_env.verify import verify_prediction


def _tool_name(action: dict[str, Any]) -> str:
    tool = action.get("tool", action.get("action"))
    return tool if isinstance(tool, str) else ""


def _ordered_segments(task: dict[str, Any]) -> list[dict[str, Any]]:
    return list(task.get("segments", []))


def _metric_summary(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    keys = [
        "score",
        "valid_schema",
        "valid_route",
        "checkpoint_reward",
        "length_ratio",
        "mean_route_distance_m",
        "num_predicted_turns",
        "num_gold_turns",
        "unknown_turn_count",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _point_in_bbox(point: dict[str, Any], bbox: list[float]) -> bool:
    west, south, east, north = bbox
    return west <= float(point["lon"]) <= east and south <= float(point["lat"]) <= north


def _bbox_center(bbox: list[float]) -> dict[str, float]:
    return {"lon": (bbox[0] + bbox[2]) / 2, "lat": (bbox[1] + bbox[3]) / 2}


def _bbox_span(bbox: list[float]) -> dict[str, float]:
    return {"lon": bbox[2] - bbox[0], "lat": bbox[3] - bbox[1]}


def _bbox_around_point(point: dict[str, Any], width: float, height: float, base: list[float]) -> list[float]:
    lon = float(point["lon"])
    lat = float(point["lat"])
    return _clamp_bbox([lon - width / 2, lat - height / 2, lon + width / 2, lat + height / 2], base)


def _clamp_bbox(bbox: list[float], base: list[float]) -> list[float]:
    west, south, east, north = bbox
    base_west, base_south, base_east, base_north = base
    width = min(east - west, base_east - base_west)
    height = min(north - south, base_north - base_south)
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2
    west = center_lon - width / 2
    east = center_lon + width / 2
    south = center_lat - height / 2
    north = center_lat + height / 2
    if west < base_west:
        east += base_west - west
        west = base_west
    if east > base_east:
        west -= east - base_east
        east = base_east
    if south < base_south:
        north += base_south - south
        south = base_south
    if north > base_north:
        south -= north - base_north
        north = base_north
    return [max(base_west, west), max(base_south, south), min(base_east, east), min(base_north, north)]


def _expanded_or_shrunk_bbox(bbox: list[float], factor: float, base: list[float]) -> list[float]:
    west, south, east, north = bbox
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2
    width = (east - west) * factor
    height = (north - south) * factor
    return _clamp_bbox(
        [
            center_lon - width / 2,
            center_lat - height / 2,
            center_lon + width / 2,
            center_lat + height / 2,
        ],
        base,
    )


def _label_number(label: str) -> int:
    try:
        return int(label.removeprefix("T"))
    except ValueError:
        return 10_000


@dataclass
class StepOutcome:
    observation: dict[str, Any]
    step_reward: float
    done: bool
    error: str | None = None


class CuaLiteMapEnv:
    """Deterministic map-tool environment backed by a RouteRL task.

    This is intentionally not browser automation. It exposes stable map panels
    and mark/unmark actions so agent traces can be scored by the existing hidden
    graph verifier.
    """

    def __init__(
        self,
        task: dict[str, Any],
        *,
        max_actions: int = 128,
        step_penalty: float = 0.0,
        invalid_action_penalty: float = -0.05,
        viewport_dir: str | Path | None = None,
        initial_view: str = "full",
        viewport_scale: float = 0.42,
        max_visible_labels: int | None = None,
        enforce_prefix_validity: bool = False,
    ) -> None:
        self.task = deepcopy(task)
        self.max_actions = max_actions
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.viewport_dir = Path(viewport_dir) if viewport_dir else None
        self.initial_view = initial_view
        self.viewport_scale = viewport_scale
        self.max_visible_labels = max_visible_labels
        self.enforce_prefix_validity = enforce_prefix_validity
        self.is_strip = self.task.get("task_type") == "route_strip"
        self.segments = _ordered_segments(self.task)
        self.segments_by_id = {segment["segment_id"]: segment for segment in self.segments}
        self.segment_ids = [segment["segment_id"] for segment in self.segments]
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.action_count = 0
        self.done = False
        self.last_error: str | None = None
        self.last_metrics: dict[str, Any] | None = None
        self.last_reward: float | None = None
        if self.is_strip:
            self.view = "overview"
            self.marked_by_segment: dict[str, list[str]] = {segment_id: [] for segment_id in self.segment_ids}
            self.marked_turns: list[str] = []
        else:
            self.view = "map"
            self.marked_by_segment = {}
            self.marked_turns = []
        self.view_bboxes = {
            self._panel_key_for_item(item): list(item["task_bbox"])
            for item in ([self.task] if not self.is_strip else self.segments)
        }
        if not self.is_strip and self.initial_view == "start":
            self._center_current_view(self.task["origin"], scale=self.viewport_scale)
        elif not self.is_strip and self.initial_view == "end":
            self._center_current_view(self.task["destination"], scale=self.viewport_scale)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        view = self._view_record()
        visible_labels = self._visible_labels()
        markable_labels = self._markable_labels(visible_labels)
        return {
            "task_id": self.task["task_id"],
            "task_type": self.task.get("task_type", "flat"),
            "view": view,
            "visible_labels": visible_labels,
            "markable_labels": markable_labels,
            "frontier_candidates": self._frontier_candidates(markable_labels),
            "visible_segments": self.segment_ids if self.is_strip and self.view == "overview" else [],
            "prediction_so_far": self.prediction(),
            "remaining_actions": max(0, self.max_actions - self.action_count),
            "done": self.done,
            "last_error": self.last_error,
            "reward": self.last_reward,
            "metrics": _metric_summary(self.last_metrics),
        }

    def prediction(self) -> dict[str, Any]:
        if self.is_strip:
            return {
                "segments": [
                    {"segment_id": segment_id, "turns": list(self.marked_by_segment[segment_id])}
                    for segment_id in self.segment_ids
                ]
            }
        return {"turns": list(self.marked_turns)}

    def step(self, action: dict[str, Any]) -> StepOutcome:
        if self.done:
            self.last_error = "episode is already done"
            return StepOutcome(self.observe(), 0.0, True, self.last_error)

        self.action_count += 1
        self.last_error = None
        step_reward = self.step_penalty
        tool = _tool_name(action)

        try:
            if tool == "observe":
                pass
            elif tool == "open_overview":
                self._open_overview()
            elif tool == "open_map":
                self._open_map()
            elif tool == "open_segment":
                self._open_segment(str(action.get("segment_id", "")))
            elif tool == "pan":
                self._pan(action)
            elif tool == "zoom_in":
                self._zoom(0.55)
            elif tool == "zoom_out":
                self._zoom(1.8)
            elif tool == "zoom_to_label":
                self._zoom_to_label(action)
            elif tool == "center_on_start":
                self._center_current_view(self._current_panel().get("origin", self.task["origin"]), scale=self.viewport_scale)
            elif tool == "center_on_end":
                self._center_current_view(
                    self._current_panel().get("destination", self.task["destination"]),
                    scale=self.viewport_scale,
                )
            elif tool == "center_on_last_mark":
                self._center_on_last_mark()
            elif tool == "pan_toward_destination":
                self._pan_toward_destination()
            elif tool == "mark":
                self._mark(action)
            elif tool == "unmark":
                self._unmark(action)
            elif tool == "clear_segment":
                self._clear_segment(str(action.get("segment_id", self.view)))
            elif tool == "finish":
                step_reward = self._finish()
            else:
                raise ValueError(f"unknown tool: {tool or '<missing>'}")
        except ValueError as exc:
            self.last_error = str(exc)
            step_reward = self.invalid_action_penalty

        if not self.done and self.action_count >= self.max_actions:
            self.last_error = self.last_error or "action budget exhausted; finishing episode"
            step_reward = self._finish()

        return StepOutcome(self.observe(), step_reward, self.done, self.last_error)

    def _view_record(self) -> dict[str, Any]:
        if self.is_strip:
            if self.view == "overview":
                return {
                    "kind": "overview",
                    "image": self.task.get("images", {}).get("overview"),
                    "segment_id": None,
                }
            segment = self.segments_by_id[self.view]
            bbox = self.view_bboxes.get(self.view, segment.get("task_bbox"))
            return {
                "kind": "segment",
                "segment_id": self.view,
                "image": self._current_image_path(segment),
                "bbox": bbox,
                "center": _bbox_center(bbox),
                "span": _bbox_span(bbox),
            }
        bbox = self.view_bboxes.get("map", self.task.get("task_bbox"))
        return {
            "kind": "map",
            "segment_id": None,
            "image": self._current_image_path(self.task),
            "bbox": bbox,
            "center": _bbox_center(bbox),
            "span": _bbox_span(bbox),
        }

    def _visible_labels(self) -> list[str]:
        return [label for label, _point in self._visible_label_items()]

    def _visible_label_items(self) -> list[tuple[str, dict[str, Any]]]:
        if self.is_strip:
            if self.view == "overview":
                return []
            segment = self.segments_by_id[self.view]
            labels = segment.get("turn_checkpoints", {})
            bbox = self.view_bboxes.get(self.view, segment["task_bbox"])
            return self._throttle_label_items(
                [(label, point) for label, point in labels.items() if _point_in_bbox(point, bbox)],
                bbox,
            )
        labels = self.task.get("turn_checkpoints", {})
        bbox = self.view_bboxes.get("map", self.task["task_bbox"])
        return self._throttle_label_items(
            [(label, point) for label, point in labels.items() if _point_in_bbox(point, bbox)],
            bbox,
        )

    def _throttle_label_items(
        self,
        items: list[tuple[str, dict[str, Any]]],
        bbox: list[float],
    ) -> list[tuple[str, dict[str, Any]]]:
        if not self.max_visible_labels or len(items) <= self.max_visible_labels:
            return sorted(items, key=lambda item: _label_number(item[0]))

        focus = self._label_focus_point(bbox)
        span = _bbox_span(bbox)
        width = max(span["lon"], 1e-9)
        height = max(span["lat"], 1e-9)

        def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, int]:
            label, point = item
            dx = (float(point["lon"]) - float(focus["lon"])) / width
            dy = (float(point["lat"]) - float(focus["lat"])) / height
            return dx * dx + dy * dy, _label_number(label)

        closest = sorted(items, key=sort_key)[: self.max_visible_labels]
        return sorted(closest, key=lambda item: _label_number(item[0]))

    def _label_focus_point(self, bbox: list[float]) -> dict[str, float]:
        last_mark = self._last_mark_point()
        if last_mark and _point_in_bbox(last_mark, bbox):
            return {"lon": float(last_mark["lon"]), "lat": float(last_mark["lat"])}
        current_panel = self._current_panel()
        origin = current_panel.get("origin", self.task.get("origin"))
        if origin and _point_in_bbox(origin, bbox):
            return {"lon": float(origin["lon"]), "lat": float(origin["lat"])}
        destination = current_panel.get("destination", self.task.get("destination"))
        if destination and _point_in_bbox(destination, bbox):
            return {"lon": float(destination["lon"]), "lat": float(destination["lat"])}
        return _bbox_center(bbox)

    def _markable_labels(self, visible_labels: list[str]) -> list[str]:
        visible = set(visible_labels)
        if self.is_strip:
            if self.view == "overview":
                return []
            marked = set(self.marked_by_segment.get(self.view, []))
        else:
            marked = set(self.marked_turns)
        labels = [label for label in visible_labels if label in visible and label not in marked]
        if self.enforce_prefix_validity and not self.is_strip:
            labels = [label for label in labels if self._prefix_valid_if_marked(label)]
        return labels

    def _prefix_valid_if_marked(self, label: str) -> bool:
        if self.is_strip:
            return True
        turns = self.marked_turns + [label]
        metrics = verify_prediction(
            self.task,
            {"task_id": self.task["task_id"], "prediction": {"turns": turns}},
        )
        return bool(metrics.get("valid_route"))

    def _frontier_candidates(self, markable_labels: list[str]) -> list[str]:
        if not markable_labels:
            return []
        panel = self._current_panel()
        checkpoints = panel.get("turn_checkpoints", {})
        if self.is_strip and self.view == "overview":
            return []
        frontier = self._last_mark_point() or panel.get("origin", self.task.get("origin"))
        if not frontier:
            return markable_labels

        def sort_key(label: str) -> tuple[float, int]:
            point = checkpoints[label]
            dx = float(point["lon"]) - float(frontier["lon"])
            dy = float(point["lat"]) - float(frontier["lat"])
            return dx * dx + dy * dy, _label_number(label)

        return sorted(markable_labels, key=sort_key)

    def _open_overview(self) -> None:
        if not self.is_strip:
            raise ValueError("open_overview is only valid for route-strip tasks")
        self.view = "overview"

    def _open_map(self) -> None:
        if self.is_strip:
            raise ValueError("open_map is only valid for flat tasks")
        self.view = "map"

    def _open_segment(self, segment_id: str) -> None:
        if not self.is_strip:
            raise ValueError("open_segment is only valid for route-strip tasks")
        if segment_id not in self.segments_by_id:
            raise ValueError(f"unknown segment_id: {segment_id}")
        self.view = segment_id

    def _panel_key_for_item(self, item: dict[str, Any]) -> str:
        if item.get("segment_id"):
            return str(item["segment_id"])
        return "map"

    def _current_panel(self) -> dict[str, Any]:
        if self.is_strip:
            if self.view == "overview":
                return self.task
            return self.segments_by_id[self.view]
        return self.task

    def _current_panel_key(self) -> str:
        if self.is_strip:
            return self.view
        return "map"

    def _current_image_path(self, panel: dict[str, Any]) -> str | None:
        if self.viewport_dir is None or (self.is_strip and self.view == "overview"):
            return panel.get("images", {}).get("map")
        key = self._current_panel_key()
        bbox = self.view_bboxes.get(key, panel.get("task_bbox"))
        safe_key = key.lower().replace("/", "_")
        out_path = self.viewport_dir / self.task["task_id"] / f"{safe_key}_{self.action_count:03d}.png"
        crop = deepcopy(panel)
        crop["task_bbox"] = list(bbox)
        crop["bbox"] = list(bbox)
        crop["turn_checkpoints"] = {label: point for label, point in self._visible_label_items()}
        render_task(crop, out_path, show_labels=True)
        return str(out_path)

    def _base_bbox_for_current_panel(self) -> list[float]:
        panel = self._current_panel()
        return list(panel["task_bbox"])

    def _pan(self, action: dict[str, Any]) -> None:
        if self.is_strip and self.view == "overview":
            raise ValueError("pan is only valid on flat maps or local segment panels")
        key = self._current_panel_key()
        bbox = self.view_bboxes[key]
        base = self._base_bbox_for_current_panel()
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        direction = str(action.get("direction", "")).lower()
        dx = float(action.get("dx", 0.0) or 0.0)
        dy = float(action.get("dy", 0.0) or 0.0)
        if direction in {"east", "right"}:
            dx += 0.45
        elif direction in {"west", "left"}:
            dx -= 0.45
        elif direction in {"north", "up"}:
            dy += 0.45
        elif direction in {"south", "down"}:
            dy -= 0.45
        elif direction in {"northwest", "north_west", "up_left"}:
            dx -= 0.45
            dy += 0.45
        elif direction in {"northeast", "north_east", "up_right"}:
            dx += 0.45
            dy += 0.45
        elif direction in {"southwest", "south_west", "down_left"}:
            dx -= 0.45
            dy -= 0.45
        elif direction in {"southeast", "south_east", "down_right"}:
            dx += 0.45
            dy -= 0.45
        elif not dx and not dy:
            raise ValueError("pan needs direction or dx/dy")
        shifted = [bbox[0] + dx * width, bbox[1] + dy * height, bbox[2] + dx * width, bbox[3] + dy * height]
        self.view_bboxes[key] = _clamp_bbox(shifted, base)

    def _zoom(self, factor: float) -> None:
        if self.is_strip and self.view == "overview":
            raise ValueError("zoom is only valid on flat maps or local segment panels")
        key = self._current_panel_key()
        self.view_bboxes[key] = _expanded_or_shrunk_bbox(
            self.view_bboxes[key],
            factor,
            self._base_bbox_for_current_panel(),
        )

    def _center_current_view(self, point: dict[str, Any], *, scale: float | None = None) -> None:
        if self.is_strip and self.view == "overview":
            raise ValueError("center tools are only valid on flat maps or local segment panels")
        key = self._current_panel_key()
        base = self._base_bbox_for_current_panel()
        if scale is None:
            span = _bbox_span(self.view_bboxes[key])
            width = span["lon"]
            height = span["lat"]
        else:
            span = _bbox_span(base)
            width = span["lon"] * scale
            height = span["lat"] * scale
        self.view_bboxes[key] = _bbox_around_point(point, width, height, base)

    def _last_mark_point(self) -> dict[str, Any] | None:
        if self.is_strip:
            if self.view == "overview":
                return None
            turns = self.marked_by_segment.get(self.view, [])
            if not turns:
                return None
            return self.segments_by_id[self.view].get("turn_checkpoints", {}).get(turns[-1])
        if not self.marked_turns:
            return None
        return self.task.get("turn_checkpoints", {}).get(self.marked_turns[-1])

    def _center_on_last_mark(self) -> None:
        point = self._last_mark_point()
        if not point:
            raise ValueError("center_on_last_mark needs at least one marked checkpoint in the current view")
        self._center_current_view(point)

    def _pan_toward_destination(self) -> None:
        if self.is_strip and self.view == "overview":
            raise ValueError("pan_toward_destination is only valid on flat maps or local segment panels")
        key = self._current_panel_key()
        bbox = self.view_bboxes[key]
        base = self._base_bbox_for_current_panel()
        destination = self._current_panel().get("destination", self.task["destination"])
        center = _bbox_center(bbox)
        span = _bbox_span(bbox)
        dx = float(destination["lon"]) - center["lon"]
        dy = float(destination["lat"]) - center["lat"]
        norm = ((dx / max(span["lon"], 1e-9)) ** 2 + (dy / max(span["lat"], 1e-9)) ** 2) ** 0.5
        if norm < 1e-6:
            return
        step = min(0.68 / norm, 1.0)
        shifted = [
            bbox[0] + dx * step,
            bbox[1] + dy * step,
            bbox[2] + dx * step,
            bbox[3] + dy * step,
        ]
        self.view_bboxes[key] = _clamp_bbox(shifted, base)

    def _zoom_to_label(self, action: dict[str, Any]) -> None:
        if self.is_strip and self.view == "overview":
            raise ValueError("zoom_to_label is only valid on local segment panels")
        label = self._turn_for_action(action)
        if label not in set(self._visible_labels()):
            raise ValueError(f"{label} is not visible in the current viewport")
        panel = self._current_panel()
        point = panel.get("turn_checkpoints", {}).get(label)
        if not point:
            raise ValueError(f"{label} is not visible on the current map")
        base = self._base_bbox_for_current_panel()
        current = self.view_bboxes[self._current_panel_key()]
        width = (current[2] - current[0]) * 0.45
        height = (current[3] - current[1]) * 0.45
        lon = float(point["lon"])
        lat = float(point["lat"])
        self.view_bboxes[self._current_panel_key()] = _clamp_bbox(
            [lon - width / 2, lat - height / 2, lon + width / 2, lat + height / 2],
            base,
        )

    def _segment_for_action(self, action: dict[str, Any]) -> str:
        segment_id = action.get("segment_id")
        if segment_id is None:
            segment_id = self.view
        segment_id = str(segment_id)
        if segment_id not in self.segments_by_id:
            raise ValueError("mark/unmark needs an open or explicit segment_id")
        return segment_id

    def _turn_for_action(self, action: dict[str, Any]) -> str:
        turn = action.get("turn", action.get("label"))
        if not isinstance(turn, str):
            raise ValueError("turn must be a string label")
        return turn

    def _mark(self, action: dict[str, Any]) -> None:
        turn = self._turn_for_action(action)
        if self.is_strip:
            segment_id = self._segment_for_action(action)
            allowed = self.segments_by_id[segment_id].get("turn_checkpoints", {})
            if turn not in allowed:
                raise ValueError(f"{turn} is not visible in {segment_id}")
            if segment_id == self.view and turn not in set(self._visible_labels()):
                raise ValueError(f"{turn} is not visible in the current {segment_id} viewport")
            if turn in self.marked_by_segment[segment_id]:
                raise ValueError(f"{turn} is already marked in {segment_id}")
            self.marked_by_segment[segment_id].append(turn)
            return

        if turn not in self.task.get("turn_checkpoints", {}):
            raise ValueError(f"{turn} is not visible on the map")
        if turn not in set(self._visible_labels()):
            raise ValueError(f"{turn} is not visible in the current viewport")
        if turn in self.marked_turns:
            raise ValueError(f"{turn} is already marked")
        if self.enforce_prefix_validity and not self._prefix_valid_if_marked(turn):
            raise ValueError(f"{turn} would make the route prefix unroutable")
        self.marked_turns.append(turn)

    def _unmark(self, action: dict[str, Any]) -> None:
        turn = self._turn_for_action(action)
        if self.is_strip:
            segment_id = self._segment_for_action(action)
            if turn in self.marked_by_segment[segment_id]:
                self.marked_by_segment[segment_id].remove(turn)
            return
        if turn in self.marked_turns:
            self.marked_turns.remove(turn)

    def _clear_segment(self, segment_id: str) -> None:
        if not self.is_strip:
            raise ValueError("clear_segment is only valid for route-strip tasks")
        if segment_id not in self.marked_by_segment:
            raise ValueError(f"unknown segment_id: {segment_id}")
        self.marked_by_segment[segment_id] = []

    def _finish(self) -> float:
        self.done = True
        self.last_metrics = verify_prediction(
            self.task,
            {"task_id": self.task["task_id"], "prediction": self.prediction()},
        )
        self.last_reward = float(self.last_metrics.get("score", 0.0))
        return self.last_reward


def load_task(tasks_path: str, *, task_id: str | None = None, task_index: int = 0) -> dict[str, Any]:
    tasks = read_jsonl(tasks_path)
    if task_id is not None:
        for task in tasks:
            if task["task_id"] == task_id:
                return task
        raise ValueError(f"task_id not found: {task_id}")
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"task_index out of range: {task_index}")
    return tasks[task_index]


def oracle_actions_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if task.get("task_type") == "route_strip":
        actions.append({"tool": "open_overview"})
        for segment in _ordered_segments(task):
            segment_id = segment["segment_id"]
            actions.append({"tool": "open_segment", "segment_id": segment_id})
            for turn in segment.get("oracle", {}).get("gold_turn_route", []):
                actions.append({"tool": "mark", "segment_id": segment_id, "turn": turn})
        actions.append({"tool": "finish"})
        return actions

    actions.append({"tool": "open_map"})
    for turn in task.get("oracle", {}).get("gold_turn_route", []):
        actions.append({"tool": "mark", "turn": turn})
    actions.append({"tool": "finish"})
    return actions


def all_visible_label_actions_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if task.get("task_type") == "route_strip":
        actions.append({"tool": "open_overview"})
        for segment in _ordered_segments(task):
            segment_id = segment["segment_id"]
            actions.append({"tool": "open_segment", "segment_id": segment_id})
            for turn in segment.get("turn_checkpoints", {}):
                actions.append({"tool": "mark", "segment_id": segment_id, "turn": turn})
        actions.append({"tool": "finish"})
        return actions

    actions.append({"tool": "open_map"})
    for turn in task.get("turn_checkpoints", {}):
        actions.append({"tool": "mark", "turn": turn})
    actions.append({"tool": "finish"})
    return actions


def empty_actions_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    if task.get("task_type") == "route_strip":
        return [{"tool": "open_overview"}, {"tool": "finish"}]
    return [{"tool": "open_map"}, {"tool": "finish"}]
