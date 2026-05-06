from __future__ import annotations

import math
from typing import Iterable, Sequence

LatLon = dict[str, float]

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(a: LatLon, b: LatLon) -> float:
    lat1 = math.radians(a["lat"])
    lat2 = math.radians(b["lat"])
    dlat = lat2 - lat1
    dlon = math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def polyline_length_m(points: Sequence[LatLon]) -> float:
    return sum(haversine_m(a, b) for a, b in zip(points, points[1:]))


def bbox_from_points(points: Sequence[LatLon], margin_m: float = 80) -> list[float]:
    if not points:
        raise ValueError("cannot build bbox from empty point list")
    min_lat = min(p["lat"] for p in points)
    max_lat = max(p["lat"] for p in points)
    min_lon = min(p["lon"] for p in points)
    max_lon = max(p["lon"] for p in points)
    mid_lat = (min_lat + max_lat) / 2
    lat_margin = margin_m / 111_320
    lon_margin = margin_m / max(1, 111_320 * math.cos(math.radians(mid_lat)))
    return [min_lon - lon_margin, min_lat - lat_margin, max_lon + lon_margin, max_lat + lat_margin]


def point_to_segment_distance_m(point: LatLon, a: LatLon, b: LatLon) -> float:
    """Approximate point-to-segment distance in meters using local equirectangular coordinates."""
    lat0 = math.radians(point["lat"])

    def project(p: LatLon) -> tuple[float, float]:
        x = math.radians(p["lon"] - point["lon"]) * math.cos(lat0) * EARTH_RADIUS_M
        y = math.radians(p["lat"] - point["lat"]) * EARTH_RADIUS_M
        return x, y

    ax, ay = project(a)
    bx, by = project(b)
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / denom))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(px, py)


def point_to_polyline_distance_m(point: LatLon, polyline: Sequence[LatLon]) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return haversine_m(point, polyline[0])
    return min(point_to_segment_distance_m(point, a, b) for a, b in zip(polyline, polyline[1:]))


def directed_hausdorff_m(a: Sequence[LatLon], b: Sequence[LatLon]) -> float:
    if not a or not b:
        return math.inf
    return max(point_to_polyline_distance_m(point, b) for point in a)


def hausdorff_distance_m(a: Sequence[LatLon], b: Sequence[LatLon]) -> float:
    return max(directed_hausdorff_m(a, b), directed_hausdorff_m(b, a))


def mean_bidirectional_distance_m(a: Sequence[LatLon], b: Sequence[LatLon]) -> float:
    if not a or not b:
        return math.inf
    da = [point_to_polyline_distance_m(point, b) for point in a]
    db = [point_to_polyline_distance_m(point, a) for point in b]
    return (sum(da) + sum(db)) / (len(da) + len(db))


def lonlat_to_latlon(point: Sequence[float]) -> LatLon:
    return {"lon": float(point[0]), "lat": float(point[1])}


def latlon_to_lonlat(point: LatLon) -> list[float]:
    return [float(point["lon"]), float(point["lat"])]


def dedupe_consecutive(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not out or out[-1] != item:
            out.append(item)
    return out

