from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx
import osmnx as ox

from route_env.geometry import bbox_from_points, haversine_m, latlon_to_lonlat


@dataclass(frozen=True)
class GenerateConfig:
    bbox: list[float]
    city: str
    network_type: str = "drive"
    min_distance_m: float = 500
    max_distance_m: float = 2000
    max_checkpoints: int = 24
    route_margin_m: float = 140
    seed: int = 7


def load_osm_graph(bbox: list[float], network_type: str = "drive") -> nx.MultiDiGraph:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    graph = ox.graph_from_bbox(tuple(bbox), network_type=network_type, simplify=True)
    return ox.distance.add_edge_lengths(graph)


def largest_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    weak = graph.to_undirected()
    component = max(nx.connected_components(weak), key=len)
    return graph.subgraph(component).copy()


def node_point(graph: nx.MultiDiGraph, node: int) -> dict[str, float]:
    data = graph.nodes[node]
    return {"lat": float(data["y"]), "lon": float(data["x"])}


def edge_geometry(graph: nx.MultiDiGraph, u: int, v: int, key: int) -> list[dict[str, float]]:
    data = graph.edges[u, v, key]
    geom = data.get("geometry")
    if geom is not None:
        return [{"lat": float(lat), "lon": float(lon)} for lon, lat in geom.coords]
    return [node_point(graph, u), node_point(graph, v)]


def shortest_path(graph: nx.MultiDiGraph, source: int, target: int) -> tuple[list[int], float]:
    path = nx.shortest_path(graph, source, target, weight="length")
    length = nx.shortest_path_length(graph, source, target, weight="length")
    return [int(n) for n in path], float(length)


def route_points(graph: nx.MultiDiGraph, route: list[int]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for u, v in zip(route, route[1:]):
        edge_key = min(graph.get_edge_data(u, v).items(), key=lambda item: item[1].get("length", 1e9))[0]
        segment = edge_geometry(graph, u, v, edge_key)
        if points:
            segment = segment[1:]
        points.extend(segment)
    if not points and route:
        points.append(node_point(graph, route[0]))
    return points


def turn_angle_deg(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> float:
    v1 = (b["lon"] - a["lon"], b["lat"] - a["lat"])
    v2 = (c["lon"] - b["lon"], c["lat"] - b["lat"])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return abs(180 - math.degrees(math.acos(cosang)))


def important_decision_nodes(graph: nx.MultiDiGraph, route: list[int], min_angle_deg: float = 30) -> list[int]:
    if len(route) <= 2:
        return route[:]
    important = [route[0]]
    for prev_node, node, next_node in zip(route, route[1:], route[2:]):
        if turn_angle_deg(node_point(graph, prev_node), node_point(graph, node), node_point(graph, next_node)) >= min_angle_deg:
            important.append(node)
        elif graph.out_degree(node) + graph.in_degree(node) >= 4:
            important.append(node)
    important.append(route[-1])
    return list(dict.fromkeys(important))


def nodes_in_bbox(graph: nx.MultiDiGraph, bbox: list[float]) -> list[int]:
    west, south, east, north = bbox
    return [
        int(n)
        for n, data in graph.nodes(data=True)
        if west <= float(data["x"]) <= east and south <= float(data["y"]) <= north
    ]


def select_checkpoints(
    graph: nx.MultiDiGraph,
    route: list[int],
    crop_nodes: list[int],
    max_checkpoints: int,
    rng: random.Random,
) -> list[int]:
    endpoint_nodes = {route[0], route[-1]}
    route_important = [node for node in important_decision_nodes(graph, route)[1:-1] if node not in endpoint_nodes]
    selected = route_important[:max_checkpoints]
    selected_set = set(selected)
    route_pts = [node_point(graph, n) for n in route]

    def dist_to_route(node: int) -> float:
        p = node_point(graph, node)
        return min(haversine_m(p, rp) for rp in route_pts)

    # Distractors should be real driving decision points, not arbitrary dots.
    candidates = [
        n
        for n in crop_nodes
        if n not in selected_set and n not in endpoint_nodes and graph.out_degree(n) + graph.in_degree(n) >= 3
    ]
    candidates.sort(key=dist_to_route)
    near = candidates[: max(max_checkpoints * 4, 40)]
    rng.shuffle(near)
    for node in near:
        if len(selected) >= max_checkpoints:
            break
        selected.append(node)
    return selected


def graph_edge_records(graph: nx.MultiDiGraph, crop_nodes: list[int]) -> list[dict[str, Any]]:
    crop = set(crop_nodes)
    seen: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if int(u) not in crop or int(v) not in crop:
            continue
        edge_id = (int(u), int(v))
        if edge_id in seen:
            continue
        seen.add(edge_id)
        records.append(
            {
                "u": str(int(u)),
                "v": str(int(v)),
                "length_m": float(data.get("length", haversine_m(node_point(graph, int(u)), node_point(graph, int(v))))),
                "geometry": [latlon_to_lonlat(p) for p in edge_geometry(graph, int(u), int(v), key)],
                "oneway": bool(data.get("oneway", False)),
                "highway": data.get("highway", ""),
            }
        )
    return records


def build_task(
    graph: nx.MultiDiGraph,
    task_id: str,
    city: str,
    network_type: str,
    bbox: list[float],
    route: list[int],
    route_length_m: float,
    max_checkpoints: int,
    route_margin_m: float,
    rng: random.Random,
) -> dict[str, Any]:
    oracle_points = route_points(graph, route)
    task_bbox = bbox_from_points(oracle_points, margin_m=route_margin_m)
    crop_nodes = nodes_in_bbox(graph, task_bbox)
    checkpoint_nodes = select_checkpoints(graph, route, crop_nodes, max_checkpoints=max_checkpoints, rng=rng)
    label_order = checkpoint_nodes[:]
    rng.shuffle(label_order)
    labels = {node: f"T{i + 1}" for i, node in enumerate(label_order)}
    origin_node = route[0]
    destination_node = route[-1]
    checkpoints = {
        labels[node]: {
            "lat": node_point(graph, node)["lat"],
            "lon": node_point(graph, node)["lon"],
            "osm_id": int(node),
        }
        for node in label_order
    }
    endpoint_nodes = {origin_node, destination_node}
    gold_turn_route = [
        labels[node] for node in important_decision_nodes(graph, route)[1:-1] if node in labels and node not in endpoint_nodes
    ]

    graph_nodes = {
        str(node): {
            "lat": node_point(graph, node)["lat"],
            "lon": node_point(graph, node)["lon"],
        }
        for node in crop_nodes
    }
    graph_edges = graph_edge_records(graph, crop_nodes)

    return {
        "task_id": task_id,
        "city": city,
        "mode": "drive",
        "network_type": network_type,
        "bbox": bbox,
        "task_bbox": task_bbox,
        "origin": {**node_point(graph, origin_node), "osm_id": int(origin_node), "label": "A"},
        "destination": {**node_point(graph, destination_node), "osm_id": int(destination_node), "label": "B"},
        "images": {"map": f"data/rendered/{task_id}/map.png"},
        "turn_checkpoints": checkpoints,
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
        "oracle": {
            "provider": "osmnx_drive",
            "distance_m": route_length_m,
            "geometry": [latlon_to_lonlat(p) for p in oracle_points],
            "gold_turn_route": gold_turn_route,
            "gold_osm_route": [int(n) for n in route],
            "turn_count": len(gold_turn_route),
        },
        "prompt": (
            "Given the real driving map image, trace a valid driving route from blue A to red B "
            "using sparse turn checkpoints. Return JSON only: {\"turns\":[\"T1\",\"T2\"]}."
        ),
    }


def generate_tasks(config: GenerateConfig, n: int) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    graph = largest_component(load_osm_graph(config.bbox, config.network_type))
    nodes = [int(n) for n in graph.nodes]
    tasks: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(500, n * 180)
    while len(tasks) < n and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(nodes, 2)
        try:
            route, length_m = shortest_path(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if not (config.min_distance_m <= length_m <= config.max_distance_m):
            continue
        if len(route) < 3:
            continue
        important = important_decision_nodes(graph, route)
        if len(important) < 3:
            continue
        task_id = f"{config.city.lower().replace(' ', '_')}_drive_{len(tasks) + 1:06d}"
        tasks.append(
            build_task(
                graph=graph,
                task_id=task_id,
                city=config.city,
                network_type=config.network_type,
                bbox=config.bbox,
                route=route,
                route_length_m=length_m,
                max_checkpoints=config.max_checkpoints,
                route_margin_m=config.route_margin_m,
                rng=rng,
            )
        )
    if len(tasks) < n:
        raise RuntimeError(f"generated {len(tasks)} tasks after {attempts} attempts; try a larger bbox or looser distances")
    return tasks
