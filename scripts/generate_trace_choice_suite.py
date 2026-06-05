#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox

from route_env.graph_tasks import build_task, important_decision_nodes, largest_component


@dataclass(frozen=True)
class BandConfig:
    name: str
    count: int
    min_distance_m: float
    max_distance_m: float
    max_checkpoints: int
    route_margin_m: float
    seed: int
    targets_per_source: int


DEFAULT_BANDS = {
    "short": BandConfig(
        name="short_500m_2km",
        count=1000,
        min_distance_m=500,
        max_distance_m=2000,
        max_checkpoints=24,
        route_margin_m=140,
        seed=7001,
        targets_per_source=5,
    ),
    "mid": BandConfig(
        name="mid_2_6km",
        count=1000,
        min_distance_m=2000,
        max_distance_m=6000,
        max_checkpoints=40,
        route_margin_m=190,
        seed=7002,
        targets_per_source=4,
    ),
    "long": BandConfig(
        name="long_8_25km",
        count=2000,
        min_distance_m=8000,
        max_distance_m=25000,
        max_checkpoints=80,
        route_margin_m=260,
        seed=7003,
        targets_per_source=3,
    ),
}


def parse_bbox(value: str) -> list[float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return parts


def graph_bbox(graph: nx.MultiDiGraph) -> list[float]:
    xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
    ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_graph(*, place: str | None, bbox: list[float] | None, network_type: str) -> tuple[nx.MultiDiGraph, list[float]]:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    if place:
        print(f"loading OSM graph for place={place!r} network_type={network_type}", file=sys.stderr, flush=True)
        graph = ox.graph_from_place(place, network_type=network_type, simplify=True, retain_all=False)
    elif bbox:
        print(f"loading OSM graph for bbox={bbox} network_type={network_type}", file=sys.stderr, flush=True)
        graph = ox.graph_from_bbox(tuple(bbox), network_type=network_type, simplify=True, retain_all=False)
    else:
        raise ValueError("either place or bbox is required")
    graph = ox.distance.add_edge_lengths(graph)
    graph = largest_component(graph)
    return graph, bbox or graph_bbox(graph)


def prepare_experiment_dir(path: Path, *, overwrite: bool) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for child in ("maps", "predictions", "results", "agent_traces", "benchmarks"):
        (path / child).mkdir(parents=True, exist_ok=True)
    tasks_path = path / "tasks.jsonl"
    if tasks_path.exists() and not overwrite:
        raise FileExistsError(f"{tasks_path} already exists; use --overwrite to replace it")
    return tasks_path


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def generate_band(
    *,
    graph: nx.MultiDiGraph,
    bbox: list[float],
    city: str,
    network_type: str,
    out_dir: Path,
    band: BandConfig,
    overwrite: bool,
    max_sources: int,
    progress_every: int,
) -> None:
    tasks_path = prepare_experiment_dir(out_dir, overwrite=overwrite)
    tmp_path = tasks_path.with_suffix(".jsonl.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rng = random.Random(band.seed)
    nodes = [int(node) for node in graph.nodes]
    used_pairs: set[tuple[int, int]] = set()
    accepted = 0
    sources_tried = 0
    started = time.time()

    print(
        f"[{band.name}] target={band.count} distance={band.min_distance_m:g}-{band.max_distance_m:g}m "
        f"max_checkpoints={band.max_checkpoints}",
        file=sys.stderr,
        flush=True,
    )

    with tmp_path.open("w", encoding="utf-8") as f:
        while accepted < band.count and sources_tried < max_sources:
            sources_tried += 1
            source = rng.choice(nodes)
            try:
                lengths, paths = nx.single_source_dijkstra(
                    graph,
                    source,
                    cutoff=band.max_distance_m,
                    weight="length",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            candidates = [
                int(target)
                for target, length_m in lengths.items()
                if target != source and band.min_distance_m <= float(length_m) <= band.max_distance_m
            ]
            rng.shuffle(candidates)
            from_this_source = 0
            for target in candidates:
                if accepted >= band.count or from_this_source >= band.targets_per_source:
                    break
                pair = (int(source), int(target))
                if pair in used_pairs:
                    continue
                route = [int(node) for node in paths[target]]
                if len(route) < 3 or len(important_decision_nodes(graph, route)) < 3:
                    continue
                task_id = f"singapore_{band.name}_{accepted + 1:06d}"
                task = build_task(
                    graph=graph,
                    task_id=task_id,
                    city=city,
                    network_type=network_type,
                    bbox=bbox,
                    route=route,
                    route_length_m=float(lengths[target]),
                    max_checkpoints=band.max_checkpoints,
                    route_margin_m=band.route_margin_m,
                    rng=rng,
                )
                task["images"]["map"] = str(out_dir / "maps" / task_id / "map.png")
                f.write(json.dumps(task, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n")
                used_pairs.add(pair)
                accepted += 1
                from_this_source += 1
                if accepted % progress_every == 0 or accepted == band.count:
                    elapsed = time.time() - started
                    rate = accepted / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{band.name}] {accepted}/{band.count} tasks "
                        f"sources={sources_tried} rate={rate:.2f}/s",
                        file=sys.stderr,
                        flush=True,
                    )

    if accepted < band.count:
        raise RuntimeError(f"[{band.name}] generated {accepted}/{band.count} after {sources_tried} source samples")

    tmp_path.replace(tasks_path)
    write_metadata(
        out_dir / "metadata.json",
        {
            "band": band.name,
            "city": city,
            "network_type": network_type,
            "bbox": bbox,
            "count": band.count,
            "min_distance_m": band.min_distance_m,
            "max_distance_m": band.max_distance_m,
            "max_checkpoints": band.max_checkpoints,
            "route_margin_m": band.route_margin_m,
            "seed": band.seed,
            "targets_per_source": band.targets_per_source,
            "sources_tried": sources_tried,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    print(f"[{band.name}] wrote {accepted} tasks to {tasks_path}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Singapore trace-choice benchmark suite.")
    parser.add_argument("--out-root", default="data/experiments/singapore_trace_choice_4k")
    parser.add_argument("--place", default="Singapore", help="OSM place query. Use empty string with --bbox.")
    parser.add_argument("--bbox", type=parse_bbox, help="Optional west,south,east,north bbox instead of --place.")
    parser.add_argument("--city", default="Singapore")
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--short-count", type=int, default=DEFAULT_BANDS["short"].count)
    parser.add_argument("--mid-count", type=int, default=DEFAULT_BANDS["mid"].count)
    parser.add_argument("--long-count", type=int, default=DEFAULT_BANDS["long"].count)
    parser.add_argument("--bands", nargs="+", choices=["short", "mid", "long"], default=["short", "mid", "long"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=25000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    place = args.place or None
    if args.bbox:
        place = None
    graph, bbox = load_graph(place=place, bbox=args.bbox, network_type=args.network_type)
    print(
        f"loaded graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} bbox={bbox}",
        file=sys.stderr,
        flush=True,
    )

    counts = {"short": args.short_count, "mid": args.mid_count, "long": args.long_count}
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_metadata(
        out_root / "metadata.json",
        {
            "city": args.city,
            "network_type": args.network_type,
            "place": place,
            "bbox": bbox,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "bands": args.bands,
        },
    )

    for name in args.bands:
        base = DEFAULT_BANDS[name]
        band = BandConfig(
            name=base.name,
            count=counts[name],
            min_distance_m=base.min_distance_m,
            max_distance_m=base.max_distance_m,
            max_checkpoints=base.max_checkpoints,
            route_margin_m=base.route_margin_m,
            seed=base.seed + args.seed_offset,
            targets_per_source=base.targets_per_source,
        )
        generate_band(
            graph=graph,
            bbox=bbox,
            city=args.city,
            network_type=args.network_type,
            out_dir=out_root / band.name,
            band=band,
            overwrite=args.overwrite,
            max_sources=args.max_sources,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
