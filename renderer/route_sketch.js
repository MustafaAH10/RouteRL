const params = new URLSearchParams(window.location.search);

function repoUrl(value, fallback = "") {
  const raw = value || fallback;
  if (!raw) return "";
  if (/^(https?:)?\/\//.test(raw) || raw.startsWith("/")) return raw;
  return `/${raw}`;
}

const taskUrl = repoUrl(params.get("tasks"), "data/experiments/singapore_benchmark_ladder_1500/easy/tasks.jsonl");

const state = {
  tasks: [],
  taskIndex: 0,
  draftTurns: [],
  selectedLabel: "",
  mode: "overview",
  preview: null,
  finished: false,
};

const svg = document.querySelector("#map");
const taskSelect = document.querySelector("#taskSelect");
const prevTask = document.querySelector("#prevTask");
const nextTask = document.querySelector("#nextTask");
const taskCounter = document.querySelector("#taskCounter");
const subtitle = document.querySelector("#subtitle");
const taskLabel = document.querySelector("#taskLabel");
const modeLabel = document.querySelector("#modeLabel");
const draftChips = document.querySelector("#draftChips");
const draftInput = document.querySelector("#draftInput");
const inspectInput = document.querySelector("#inspectInput");
const inspectStatus = document.querySelector("#inspectStatus");
const previewMetrics = document.querySelector("#previewMetrics");
const toolLog = document.querySelector("#toolLog");
const debugGoldToggle = document.querySelector("#debugGoldToggle");

document.querySelector("#overviewTool").addEventListener("click", () => observeOverview());
document.querySelector("#inspectTool").addEventListener("click", () => inspectTarget(inspectInput.value.trim()));
document.querySelector("#previewTool").addEventListener("click", () => previewDraft());
document.querySelector("#finishTool").addEventListener("click", () => finishDraft());
document.querySelector("#applyDraft").addEventListener("click", () => applyDraftInput());
document.querySelector("#clearDraft").addEventListener("click", () => editRoute([]));
debugGoldToggle.addEventListener("change", draw);
prevTask.addEventListener("click", () => setTask(state.taskIndex - 1));
nextTask.addEventListener("click", () => setTask(state.taskIndex + 1));
taskSelect.addEventListener("change", () => setTask(Number(taskSelect.value)));

async function readJsonl(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${url}: ${response.status}`);
  const text = await response.text();
  return text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function selectedTask() {
  return state.tasks[state.taskIndex];
}

function setTask(index) {
  if (!state.tasks.length) return;
  state.taskIndex = Math.max(0, Math.min(state.tasks.length - 1, index));
  taskSelect.value = String(state.taskIndex);
  state.draftTurns = [];
  state.selectedLabel = "";
  state.preview = null;
  state.finished = false;
  inspectInput.value = "";
  observeOverview(false);
}

function taskIndexFromQuery() {
  const requested = params.get("task");
  if (!requested) return 0;
  const numeric = Number(requested);
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < state.tasks.length) return numeric;
  const oneBased = numeric - 1;
  if (Number.isInteger(oneBased) && oneBased >= 0 && oneBased < state.tasks.length) return oneBased;
  const byId = state.tasks.findIndex((task) => task.task_id === requested);
  return byId >= 0 ? byId : 0;
}

function setupTaskSelect() {
  taskSelect.replaceChildren(
    ...state.tasks.map((task, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      const km = Number(task.oracle?.distance_m || 0) / 1000;
      option.textContent = `${index + 1}. ${task.task_id} · ${km.toFixed(2)} km`;
      return option;
    }),
  );
}

function observeOverview(log = true) {
  state.mode = "overview";
  state.selectedLabel = "";
  state.finished = false;
  if (log) appendLog("observe_overview()");
  draw();
}

function inspectTarget(raw) {
  const target = raw || state.selectedLabel;
  const task = selectedTask();
  if (!target || !task) {
    inspectStatus.textContent = "Select a checkpoint or enter a grid cell like B2.";
    return;
  }
  state.mode = "inspect";
  state.selectedLabel = target.toUpperCase();
  inspectInput.value = state.selectedLabel;
  appendLog(`inspect("${state.selectedLabel}")`);
  draw();
}

function applyDraftInput() {
  const parsed = parseTurns(draftInput.value);
  editRoute(parsed);
}

function editRoute(turns) {
  const allowed = new Set(Object.keys(selectedTask().turn_checkpoints || {}));
  const cleaned = [];
  for (const turn of turns) {
    const label = String(turn).toUpperCase();
    if (allowed.has(label) && cleaned[cleaned.length - 1] !== label) cleaned.push(label);
  }
  state.draftTurns = cleaned;
  state.preview = null;
  state.finished = false;
  appendLog(`edit_route(${JSON.stringify({ turns: state.draftTurns })})`);
  syncDraftUi();
  draw();
}

function parseTurns(value) {
  const text = value.trim();
  if (!text) return [];
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
    if (Array.isArray(parsed.turns)) return parsed.turns;
  } catch {
    return text
      .split(/[,\s]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function previewDraft() {
  state.preview = buildPreview(selectedTask(), state.draftTurns);
  state.finished = false;
  appendLog(`preview_route(${JSON.stringify({ turns: state.draftTurns })}) -> ${state.preview.valid ? "valid" : "invalid"}`);
  draw();
}

function finishDraft() {
  state.preview = buildPreview(selectedTask(), state.draftTurns);
  state.finished = true;
  appendLog(`finish(${JSON.stringify({ turns: state.draftTurns })}) -> score ${state.preview.score.toFixed(3)}`);
  draw();
}

function appendTurn(label) {
  if (!state.draftTurns.includes(label)) {
    editRoute([...state.draftTurns, label]);
  } else {
    state.selectedLabel = label;
    inspectInput.value = label;
    draw();
  }
}

function removeTurn(index) {
  const next = state.draftTurns.slice();
  next.splice(index, 1);
  editRoute(next);
}

function syncDraftUi() {
  draftInput.value = JSON.stringify(state.draftTurns);
  draftChips.replaceChildren(
    ...state.draftTurns.map((turn, index) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = turn;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "x";
      button.addEventListener("click", () => removeTurn(index));
      chip.append(button);
      return chip;
    }),
  );
}

function graphFor(task) {
  const nodes = new Map(Object.entries(task.graph.nodes || {}).map(([id, point]) => [String(id), point]));
  const adjacency = new Map();
  for (const edge of task.graph.edges || []) {
    const u = String(edge.u);
    if (!adjacency.has(u)) adjacency.set(u, []);
    adjacency.get(u).push({
      u,
      v: String(edge.v),
      length: Number(edge.length_m || 0),
      geometry: edge.geometry || [],
    });
  }
  return { nodes, adjacency };
}

function dijkstra(graph, start, end) {
  const dist = new Map([[start, 0]]);
  const prev = new Map();
  const queue = [{ node: start, distance: 0 }];
  while (queue.length) {
    queue.sort((a, b) => a.distance - b.distance);
    const current = queue.shift();
    if (!current || current.distance !== dist.get(current.node)) continue;
    if (current.node === end) break;
    for (const edge of graph.adjacency.get(current.node) || []) {
      const nextDistance = current.distance + edge.length;
      if (nextDistance < (dist.get(edge.v) ?? Infinity)) {
        dist.set(edge.v, nextDistance);
        prev.set(edge.v, { node: current.node, edge });
        queue.push({ node: edge.v, distance: nextDistance });
      }
    }
  }
  if (!dist.has(end)) return null;
  const edges = [];
  const nodes = [end];
  let cursor = end;
  while (cursor !== start) {
    const item = prev.get(cursor);
    if (!item) return null;
    edges.push(item.edge);
    cursor = item.node;
    nodes.push(cursor);
  }
  edges.reverse();
  nodes.reverse();
  return { nodes, edges, distance: dist.get(end) };
}

function buildPreview(task, turns) {
  if (!turns.length) {
    return {
      valid: false,
      turns: [],
      waypointNodes: [],
      routeNodes: [],
      geometry: [],
      distance: 0,
      lengthRatio: Infinity,
      invalidSegments: [],
      coverage: 0,
      precision: 0,
      order: 0,
      score: 0,
    };
  }
  const graph = graphFor(task);
  const checkpoints = task.turn_checkpoints || {};
  const origin = String(task.origin.osm_id);
  const destination = String(task.destination.osm_id);
  const waypointNodes = [
    origin,
    ...turns.map((turn) => checkpoints[turn]).filter(Boolean).map((point) => String(point.osm_id)),
    destination,
  ];
  const allEdges = [];
  const allNodes = [];
  let totalDistance = 0;
  const invalidSegments = [];
  for (let index = 0; index < waypointNodes.length - 1; index += 1) {
    const start = waypointNodes[index];
    const end = waypointNodes[index + 1];
    const segment = dijkstra(graph, start, end);
    if (!segment) {
      invalidSegments.push([start, end]);
      continue;
    }
    totalDistance += segment.distance;
    allEdges.push(...segment.edges);
    if (!allNodes.length) allNodes.push(...segment.nodes);
    else allNodes.push(...segment.nodes.slice(1));
  }
  const geometry = edgesToGeometry(allEdges);
  const valid = invalidSegments.length === 0 && waypointNodes.length >= 2;
  const goldTurns = task.oracle?.gold_turn_route || [];
  const alignment = checkpointAlignment(turns, goldTurns);
  const oracleDistance = Number(task.oracle?.distance_m || 0);
  const lengthRatio = valid && oracleDistance > 0 ? totalDistance / oracleDistance : Infinity;
  const distanceReward = Number.isFinite(lengthRatio) && lengthRatio > 0 ? Math.exp(-Math.abs(Math.log(lengthRatio))) : 0;
  const endpointReward = valid ? 1 : 0;
  const score = Math.max(0, Math.min(1, 0.25 * endpointReward + 0.35 * distanceReward + 0.4 * alignment.reward));
  return {
    valid,
    turns: turns.slice(),
    waypointNodes,
    routeNodes: allNodes,
    geometry,
    distance: totalDistance,
    lengthRatio,
    invalidSegments,
    coverage: alignment.coverage,
    precision: alignment.precision,
    order: alignment.order,
    score,
  };
}

function edgesToGeometry(edges) {
  const out = [];
  for (const edge of edges) {
    const points = edge.geometry || [];
    if (!points.length) continue;
    if (out.length) out.push(...points.slice(1));
    else out.push(...points);
  }
  return out;
}

function checkpointAlignment(turns, goldTurns) {
  if (!goldTurns.length) {
    const reward = turns.length ? 0.5 : 1;
    return { reward, coverage: reward, precision: reward, order: reward };
  }
  const unique = new Set(turns);
  const gold = new Set(goldTurns);
  const overlap = [...unique].filter((turn) => gold.has(turn)).length;
  const coverage = overlap / gold.size;
  const precision = turns.length ? overlap / unique.size : 0;
  const order = lcs(turns, goldTurns) / goldTurns.length;
  const reward = 0.45 * coverage + 0.35 * order + 0.2 * precision;
  return { reward, coverage, precision, order };
}

function lcs(a, b) {
  const previous = Array(b.length + 1).fill(0);
  for (const left of a) {
    let diagonal = 0;
    for (let j = 0; j < b.length; j += 1) {
      const saved = previous[j + 1];
      previous[j + 1] = left === b[j] ? diagonal + 1 : Math.max(previous[j + 1], previous[j]);
      diagonal = saved;
    }
  }
  return previous[b.length];
}

function lonLat(point) {
  if (Array.isArray(point)) return point;
  return [Number(point.lon), Number(point.lat)];
}

function projectFactory(bbox, width, height) {
  const [west, south, east, north] = bbox;
  return ([lon, lat]) => [
    ((lon - west) / (east - west || 1)) * width,
    ((north - lat) / (north - south || 1)) * height,
  ];
}

function pathData(points, project) {
  return points
    .map((point, index) => {
      const [x, y] = project(lonLat(point));
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function svgEl(name, attrs = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

function edgeClass(edge) {
  const highway = Array.isArray(edge.highway) ? edge.highway[0] : edge.highway;
  if (["motorway", "trunk", "primary", "secondary"].includes(highway)) return "major";
  if (["tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link"].includes(highway)) return "medium";
  return "minor";
}

function bboxForMode(task, width, height) {
  if (state.mode !== "inspect" || !state.selectedLabel) return [0, 0, width, height];
  const project = projectFactory(task.task_bbox, width, height);
  const target = checkpointPoint(task, state.selectedLabel) || gridCellCenter(task.task_bbox, state.selectedLabel);
  if (!target) return [0, 0, width, height];
  const [x, y] = project([target.lon, target.lat]);
  const size = Math.max(120, Math.min(width, height) * 0.38);
  return [
    Math.max(0, x - size / 2),
    Math.max(0, y - size / 2),
    Math.min(width, x + size / 2),
    Math.min(height, y + size / 2),
  ];
}

function checkpointPoint(task, label) {
  return task.turn_checkpoints?.[label.toUpperCase()] || null;
}

function gridCellCenter(bbox, cell) {
  const match = /^([A-D])([1-4])$/i.exec(cell);
  if (!match) return null;
  const col = match[1].toUpperCase().charCodeAt(0) - "A".charCodeAt(0);
  const row = Number(match[2]) - 1;
  const [west, south, east, north] = bbox;
  return {
    lon: west + ((col + 0.5) / 4) * (east - west),
    lat: north - ((row + 0.5) / 4) * (north - south),
  };
}

function draw() {
  const task = selectedTask();
  if (!task) return;
  const width = 1000;
  const height = 760;
  const project = projectFactory(task.task_bbox, width, height);
  const viewBox = bboxForMode(task, width, height);
  svg.setAttribute("viewBox", viewBox.join(" "));
  svg.replaceChildren();

  const defs = svgEl("defs");
  defs.append(
    svgEl("marker", {
      id: "arrow",
      markerWidth: "8",
      markerHeight: "8",
      refX: "6",
      refY: "3",
      orient: "auto",
      markerUnits: "strokeWidth",
    }),
  );
  defs.querySelector("marker").append(svgEl("path", { d: "M0,0 L0,6 L7,3 z", fill: "#3b82f6", opacity: "0.75" }));
  svg.append(defs);

  drawGrid(svg, project, task.task_bbox, width, height);
  const roads = svgEl("g");
  for (const edge of task.graph.edges || []) {
    roads.append(
      svgEl("path", {
        d: pathData(edge.geometry || [], project),
        class: `road ${edgeClass(edge)}`,
        fill: "none",
        "marker-mid": edge.oneway ? "url(#arrow)" : "",
      }),
    );
  }
  svg.append(roads);

  if (debugGoldToggle.checked && task.oracle?.geometry?.length) {
    svg.append(svgEl("path", { d: pathData(task.oracle.geometry, project), class: "gold-route" }));
  }

  if (state.preview?.geometry?.length) {
    svg.append(svgEl("path", { d: pathData(state.preview.geometry, project), class: "draft-route" }));
  }

  drawMarker(svg, project, task.origin, "A", "#1664d9");
  drawMarker(svg, project, task.destination, "B", "#dc2626");
  drawCheckpoints(svg, project, task);
  updateChrome(task);
}

function drawGrid(parent, project, bbox) {
  const [west, south, east, north] = bbox;
  const group = svgEl("g");
  for (let i = 1; i < 4; i += 1) {
    const xLon = west + (i / 4) * (east - west);
    const [x] = project([xLon, south]);
    group.append(svgEl("line", { x1: x, y1: 0, x2: x, y2: 760, class: "grid-line" }));
  }
  for (let i = 1; i < 4; i += 1) {
    const yLat = south + (i / 4) * (north - south);
    const [, y] = project([west, yLat]);
    group.append(svgEl("line", { x1: 0, y1: y, x2: 1000, y2: y, class: "grid-line" }));
  }
  for (let col = 0; col < 4; col += 1) {
    const xLon = west + ((col + 0.5) / 4) * (east - west);
    const [x] = project([xLon, north]);
    group.append(svgEl("text", { x, y: 18, class: "grid-label", "text-anchor": "middle" }, String.fromCharCode(65 + col)));
  }
  for (let row = 0; row < 4; row += 1) {
    const yLat = north - ((row + 0.5) / 4) * (north - south);
    const [, y] = project([west, yLat]);
    group.append(svgEl("text", { x: 10, y, class: "grid-label" }, String(row + 1)));
  }
  parent.append(group);
}

function drawMarker(parent, project, point, label, color) {
  const [x, y] = project([point.lon, point.lat]);
  const group = svgEl("g", { class: "marker" });
  group.append(svgEl("circle", { cx: x, cy: y, r: 15, fill: color }));
  group.append(svgEl("text", { x, y: y + 0.5, "font-size": "14" }, label));
  parent.append(group);
}

function drawCheckpoints(parent, project, task) {
  const group = svgEl("g");
  const draft = new Set(state.draftTurns);
  for (const [label, point] of Object.entries(task.turn_checkpoints || {})) {
    const [x, y] = project([point.lon, point.lat]);
    const node = svgEl("g", {
      class: `checkpoint ${label === state.selectedLabel ? "selected" : ""} ${draft.has(label) ? "in-draft" : ""}`,
      tabindex: "0",
    });
    node.append(svgEl("circle", { cx: x, cy: y, r: 12 }));
    node.append(svgEl("text", { x, y: y + 0.5, "font-size": "9" }, label));
    node.addEventListener("click", () => {
      state.selectedLabel = label;
      inspectInput.value = label;
      appendTurn(label);
    });
    node.addEventListener("dblclick", () => inspectTarget(label));
    group.append(node);
  }
  parent.append(group);
}

function metric(label, value) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  return [dt, dd];
}

function updateChrome(task) {
  const km = Number(task.oracle?.distance_m || 0) / 1000;
  subtitle.textContent = `${state.tasks.length} tasks loaded from ${taskUrl}`;
  taskLabel.textContent = `${task.task_id} · oracle ${km.toFixed(2)} km`;
  taskCounter.textContent = `${state.taskIndex + 1} / ${state.tasks.length}`;
  prevTask.disabled = state.taskIndex <= 0;
  nextTask.disabled = state.taskIndex >= state.tasks.length - 1;
  modeLabel.textContent = state.mode === "inspect" ? `Inspect ${state.selectedLabel}` : "Overview";
  inspectStatus.textContent =
    state.mode === "inspect"
      ? `Inspecting ${state.selectedLabel}. Double-click another checkpoint or use observe_overview.`
      : "Click a checkpoint to append it. Double-click or type a label/grid cell to inspect.";
  syncDraftUi();
  updateMetrics(task);
}

function updateMetrics(task) {
  const preview = state.preview || buildPreview(task, state.draftTurns);
  const rows = [
    metric("draft turns", String(state.draftTurns.length)),
    metric("preview valid", preview.valid ? "true" : "false"),
    metric("distance", preview.valid ? `${preview.distance.toFixed(1)} m` : "invalid"),
    metric("length ratio", Number.isFinite(preview.lengthRatio) ? preview.lengthRatio.toFixed(3) : "inf"),
    metric("coverage", preview.coverage.toFixed(3)),
    metric("order", preview.order.toFixed(3)),
    metric(state.finished ? "final score" : "preview score", preview.score.toFixed(3)),
  ];
  previewMetrics.replaceChildren(...rows.flat());
}

function appendLog(line) {
  const timestamp = new Date().toLocaleTimeString();
  toolLog.textContent = `[${timestamp}] ${line}\n${toolLog.textContent}`.slice(0, 4000);
}

async function main() {
  state.tasks = await readJsonl(taskUrl);
  setupTaskSelect();
  state.taskIndex = taskIndexFromQuery();
  taskSelect.value = String(state.taskIndex);
  appendLog("observe_overview()");
  draw();
}

main().catch((error) => {
  subtitle.textContent = error.message;
  toolLog.textContent = error.stack || String(error);
});
