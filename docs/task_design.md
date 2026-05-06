# RouteRL-Drive Task Design

RouteRL should now be driving-first. Walking can be added later as a separate
mode, but the immediate benchmark should focus on driving routes, one-way
constraints, road hierarchy, and long-route context.

## Why Driving-First

Driving routes need different information than walking routes:

- one-way direction matters;
- highways and ramps matter;
- minor roads may connect in subtle ways;
- divided roads and slip roads create local ambiguity;
- pedestrian-only paths should be excluded;
- turn restrictions can be added later.

Use OSM driving graphs as the hidden environment:

```text
network_type = drive
```

The graph is directed, so one-way constraints are represented in the verifier.

## Problem With Dense Labels

The current dense `N1...N60` map is only a smoke test. It creates a visual OCR
task instead of a clean route-planning task.

Problems:

- labels overlap;
- minor roads become unreadable;
- VLMs spend capacity reading tiny text;
- output sequences become too long;
- the setup does not resemble real navigation maps.

## Better Local Driving Task

For short and medium trips, use one canonical top-down driving map plus optional
zoom panels.

Model input:

```text
canonical driving map
start marker A
destination marker B
drivable roads styled by hierarchy
one-way arrows
sparse turn checkpoints T1...T20
optional start/destination zooms
```

Model output:

```json
{
  "turns": ["T3", "T8", "T12"],
  "confidence": 0.7
}
```

Verifier:

```text
A -> T3 -> T8 -> T12 -> B
```

is routed through the hidden directed OSM driving graph and compared to the
hidden optimal route.

## Better Long-Route Task

For cross-island or multi-kilometer routes, do not use one giant image. Use a
route strip.

Input bundle:

```text
Panel 0: overview corridor map
Panel 1: segment 1 local driving map
Panel 2: segment 2 local driving map
Panel 3: segment 3 local driving map
...
```

Each segment panel has its own local markers:

```text
S1_A -> S1_B
S2_A -> S2_B
S3_A -> S3_B
```

Output:

```json
{
  "segments": [
    {"segment_id": "S1", "turns": ["T2", "T5"]},
    {"segment_id": "S2", "turns": ["T1", "T4", "T8"]},
    {"segment_id": "S3", "turns": ["T3"]}
  ],
  "confidence": 0.66
}
```

This gives the VLM enough local detail while preserving global route context.

## Reward For RL

Local segment reward:

```text
valid JSON/schema
known checkpoint IDs
directed route exists through checkpoints
starts near segment A
ends near segment B
length close to segment oracle
geometry close to segment oracle
few/no loops
reasonable checkpoint count
```

Full route-strip reward:

```text
average segment reward
+ stitched directed route validity
+ full-route length ratio reward
+ full-route geometry similarity reward
+ corridor plausibility reward
```

This gives denser feedback than a single long-route score.

## Visual Styling

Driving map styling should make road constraints visually legible:

```text
expressways / major roads: thick, high-contrast
arterial roads: medium stroke
minor roads: thin but visible
one-way roads: repeated arrows
ramps/slip roads: distinct color or stroke
non-drivable paths: hidden or faint
checkpoints: sparse, large enough to read
A/B markers: large and high contrast
```

Avoid dumping the graph as text. The model should use the visual map, not an
adjacency list.

## Streets-GL

Use Streets-GL for visualization and demos:

```text
predicted route geometry -> Streets-GL render
Google/OSRM comparison route -> Streets-GL render
```

Do not make Streets-GL the first canonical coordinate frame. The top-down 2D map
is easier to score and align with OSM graph geometry.

