const params = new URLSearchParams(window.location.search);

function repoUrl(value, fallback = "") {
  const raw = value || fallback;
  if (!raw) return "";
  if (/^(https?:)?\/\//.test(raw) || raw.startsWith("/")) return raw;
  return `/${raw}`;
}

const taskUrl = repoUrl(params.get("tasks"), "data/experiments/long_8_25km_route_strip_probe/tasks.jsonl");
const predictionUrl = repoUrl(params.get("predictions"));
const resultUrl = repoUrl(params.get("results"));

const svg = document.querySelector("#map");
const taskSelect = document.querySelector("#taskSelect");
const prevTaskButton = document.querySelector("#prevTask");
const nextTaskButton = document.querySelector("#nextTask");
const taskCounter = document.querySelector("#taskCounter");
const panelSelect = document.querySelector("#panelSelect");
const panelField = document.querySelector("#panelField");
const subtitle = document.querySelector("#subtitle");
const metrics = document.querySelector("#metrics");
const toggles = {
  graph: document.querySelector("#graphToggle"),
  arrows: document.querySelector("#arrowsToggle"),
  oracle: document.querySelector("#oracleToggle"),
  agent: document.querySelector("#agentToggle"),
  checkpoints: document.querySelector("#checkpointsToggle"),
};

const state = {
  tasks: [],
  predictions: new Map(),
  results: new Map(),
  view: {
    x: 0,
    y: 0,
    k: 1,
    width: 1000,
    height: 700,
  },
  viewport: null,
  dragging: null,
};

const zoomLimits = { min: 0.65, max: 12 };

async function readJsonl(url, optional = false) {
  if (!url) return [];
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    if (optional) return [];
    throw new Error(`Could not load ${url}: ${response.status}`);
  }
  const text = await response.text();
  return text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map(parseJsonLine);
}

function parseJsonLine(line) {
  try {
    return JSON.parse(line);
  } catch (error) {
    const strict = line.replace(/([:\[,])\s*(NaN|-?Infinity)\s*(?=[,\]}])/g, "$1 null");
    return JSON.parse(strict);
  }
}

function projectFactory(bbox, width, height) {
  const [west, south, east, north] = bbox;
  return ([lon, lat]) => [
    ((lon - west) / (east - west || 1)) * width,
    ((north - lat) / (north - south || 1)) * height,
  ];
}

function pathData(points, project) {
  if (!points || points.length === 0) return "";
  return points
    .map((point, index) => {
      const [x, y] = project(point);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function el(name, attrs = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

function metric(label, value) {
  const row = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  row.append(dt, dd);
  return row;
}

function selectedTask() {
  return state.tasks[selectedTaskIndex()];
}

function selectedTaskIndex() {
  if (!state.tasks.length) return 0;
  const value = Number(taskSelect.value);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(state.tasks.length - 1, value));
}

function selectedPanel(task) {
  if (task.task_type !== "route_strip") return task;
  const value = panelSelect.value || "overview";
  if (value === "overview") return task;
  return task.segments.find((segment) => segment.segment_id === value) || task;
}

function resetView() {
  state.view.x = 0;
  state.view.y = 0;
  state.view.k = 1;
}

function applyViewportTransform() {
  if (!state.viewport) return;
  state.viewport.setAttribute("transform", `translate(${state.view.x} ${state.view.y}) scale(${state.view.k})`);
}

function svgPoint(event) {
  const rect = svg.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * state.view.width,
    y: ((event.clientY - rect.top) / rect.height) * state.view.height,
  };
}

function taskLabel(task, index) {
  const distanceKm = Number(task.oracle?.distance_m || 0) / 1000;
  const segments = task.task_type === "route_strip" ? ` · ${task.segments.length} seg` : "";
  return `Task ${index + 1} · ${distanceKm.toFixed(1)} km${segments} · ${task.task_id}`;
}

function taskIndexFromQuery() {
  const requested = params.get("task");
  if (!requested) return 0;
  const numeric = Number(requested);
  if (Number.isInteger(numeric)) {
    const oneBased = numeric - 1;
    if (oneBased >= 0 && oneBased < state.tasks.length) return oneBased;
    if (numeric >= 0 && numeric < state.tasks.length) return numeric;
  }
  const byId = state.tasks.findIndex((task) => task.task_id === requested || task.source_task_id === requested);
  return byId >= 0 ? byId : 0;
}

function updateUrlSelection() {
  const task = selectedTask();
  if (!task) return;
  const next = new URL(window.location.href);
  next.searchParams.set("task", task.task_id);
  if (panelSelect.value && panelSelect.value !== "overview") {
    next.searchParams.set("panel", panelSelect.value);
  } else {
    next.searchParams.delete("panel");
  }
  window.history.replaceState(null, "", next);
}

function updateTaskNav() {
  const index = selectedTaskIndex();
  const total = state.tasks.length;
  taskCounter.textContent = total ? `Task ${index + 1} / ${total}` : "Task 0 / 0";
  prevTaskButton.disabled = index <= 0;
  nextTaskButton.disabled = index >= total - 1;
}

function setTaskIndex(index) {
  if (!state.tasks.length) return;
  const clamped = Math.max(0, Math.min(state.tasks.length - 1, index));
  taskSelect.value = String(clamped);
  resetView();
  setupPanelSelector();
  updateUrlSelection();
  draw();
}

function setupSelectors() {
  taskSelect.replaceChildren(
    ...state.tasks.map((task, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = taskLabel(task, index);
      return option;
    }),
  );
  taskSelect.value = String(taskIndexFromQuery());
  taskSelect.addEventListener("change", () => {
    setTaskIndex(selectedTaskIndex());
  });
  prevTaskButton.addEventListener("click", () => setTaskIndex(selectedTaskIndex() - 1));
  nextTaskButton.addEventListener("click", () => setTaskIndex(selectedTaskIndex() + 1));
  panelSelect.addEventListener("change", () => {
    resetView();
    updateUrlSelection();
    draw();
  });
  for (const toggle of Object.values(toggles)) toggle.addEventListener("change", draw);
  setupPanelSelector({ useQueryPanel: true });
}

function setupPanelSelector({ useQueryPanel = false } = {}) {
  const task = selectedTask();
  panelSelect.replaceChildren();
  if (!task || task.task_type !== "route_strip") {
    panelField.hidden = true;
    return;
  }
  panelField.hidden = false;
  const overview = document.createElement("option");
  overview.value = "overview";
  overview.textContent = "Overview";
  panelSelect.append(overview);
  for (const segment of task.segments) {
    const option = document.createElement("option");
    option.value = segment.segment_id;
    option.textContent = `${segment.segment_id} local map`;
    panelSelect.append(option);
  }
  const requestedPanel = useQueryPanel ? params.get("panel") : null;
  if (requestedPanel && [...panelSelect.options].some((option) => option.value === requestedPanel)) {
    panelSelect.value = requestedPanel;
  }
}

function drawEdges(panel, project, scale) {
  const group = el("g");
  if (!toggles.graph.checked) return group;
  for (const edge of panel.graph?.edges || []) {
    const path = el("path", {
      d: pathData(edge.geometry, project),
      class: "road",
      "stroke-width": String(Math.max(0.7, (edge.highway || "").includes("motorway") ? 2.8 : 1.4) * scale),
    });
    group.append(path);
    if (toggles.arrows.checked && edge.oneway) group.append(drawEdgeArrow(edge, project, scale));
  }
  return group;
}

function drawEdgeArrow(edge, project, scale) {
  const points = edge.geometry || [];
  if (points.length < 2) return el("g");
  const mid = Math.max(1, Math.floor(points.length / 2));
  const [x0, y0] = project(points[mid - 1]);
  const [x1, y1] = project(points[mid]);
  if (x0 === x1 && y0 === y1) return el("g");
  const angle = (Math.atan2(y1 - y0, x1 - x0) * 180) / Math.PI;
  const size = 5.8 * scale;
  return el("path", {
    d: `M${(-size).toFixed(2)} ${(-size * 0.62).toFixed(2)} L${size.toFixed(2)} 0 L${(-size).toFixed(2)} ${(size * 0.62).toFixed(2)} Z`,
    class: "road-arrow",
    transform: `translate(${x1.toFixed(2)} ${y1.toFixed(2)}) rotate(${angle.toFixed(2)})`,
  });
}

function labelNumber(label) {
  const value = Number(String(label).replace(/^T/, ""));
  return Number.isFinite(value) ? value : 1_000_000;
}

function checkpointEntries(task, panel) {
  if (task.task_type === "route_strip" && panel === task) {
    return task.segments.flatMap((segment) =>
      Object.entries(segment.turn_checkpoints || {}).map(([label, point]) => ({
        label,
        point,
        segmentId: segment.segment_id,
      })),
    );
  }
  return Object.entries(panel.turn_checkpoints || {}).map(([label, point]) => ({ label, point, segmentId: panel.segment_id }));
}

function intersects(a, b) {
  return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
}

function labelOffset(x, y, label, placed, width, height) {
  const candidates = [
    [7, -7],
    [9, 12],
    [-42, -7],
    [-42, 12],
    [12, -22],
    [-48, -22],
    [12, 28],
    [-48, 28],
    [32, 0],
    [-66, 0],
    [0, -42],
    [0, 48],
  ];
  const labelWidth = Math.max(30, label.length * 8 + 14);
  const labelHeight = 16;
  let best = candidates[0];
  let bestCost = Infinity;
  for (const [dx, dy] of candidates) {
    const box = { x0: x + dx, y0: y + dy - labelHeight, x1: x + dx + labelWidth, y1: y + dy + 3 };
    const overlap = placed.reduce((count, other) => count + (intersects(box, other) ? 1 : 0), 0);
    const outside =
      Math.max(0, -box.x0) + Math.max(0, box.x1 - width) + Math.max(0, -box.y0) + Math.max(0, box.y1 - height);
    const cost = overlap * 1000 + outside * 20 + Math.hypot(dx, dy);
    if (cost < bestCost) {
      bestCost = cost;
      best = [dx, dy];
    }
  }
  const [dx, dy] = best;
  placed.push({ x0: x + dx, y0: y + dy - labelHeight, x1: x + dx + labelWidth, y1: y + dy + 3 });
  return best;
}

function drawCheckpoints(task, panel, project, width, height) {
  const group = el("g");
  if (!toggles.checkpoints.checked) return group;
  const entries = checkpointEntries(task, panel).sort((a, b) => labelNumber(a.label) - labelNumber(b.label));
  const placed = [];
  for (const { label, point } of entries) {
    const [x, y] = project([point.lon, point.lat]);
    group.append(el("circle", { cx: x, cy: y, r: 4.2, fill: "#111", stroke: "#fff", "stroke-width": 1.3 }));
    placed.push({ x0: x - 5, y0: y - 5, x1: x + 5, y1: y + 5 });
  }
  for (const { label, point } of entries) {
    const [x, y] = project([point.lon, point.lat]);
    const [dx, dy] = labelOffset(x, y, label, placed, width, height);
    group.append(el("text", { x: x + dx, y: y + dy, class: "checkpoint-label" }, label));
  }
  return group;
}

function drawMarkers(panel, project) {
  const group = el("g");
  for (const [key, color, label] of [
    ["origin", "#1664d9", "A"],
    ["destination", "#d92525", "B"],
  ]) {
    const point = panel[key];
    if (!point) continue;
    const [x, y] = project([point.lon, point.lat]);
    group.append(el("circle", { cx: x, cy: y, r: 11, fill: color, stroke: "#fff", "stroke-width": 2 }));
    group.append(el("text", { x, y: y + 4, "text-anchor": "middle", fill: "#fff", "font-size": 12, "font-weight": 800 }, label));
  }
  return group;
}

function drawSegmentBoxes(task, project) {
  const group = el("g");
  if (task.task_type !== "route_strip" || panelSelect.value !== "overview") return group;
  for (const segment of task.segments) {
    const [west, south, east, north] = segment.task_bbox;
    const [x0, y0] = project([west, north]);
    const [x1, y1] = project([east, south]);
    group.append(el("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, class: "segment-box" }));
    group.append(el("text", { x: x0 + 6, y: y0 + 16, class: "segment-label" }, segment.segment_id));
  }
  return group;
}

function drawRoutes(task, panel, project) {
  const group = el("g");
  const result = state.results.get(task.task_id);
  if (toggles.oracle.checked && panel.oracle?.geometry) {
    group.append(el("path", { d: pathData(panel.oracle.geometry, project), class: "route-oracle" }));
  } else if (toggles.oracle.checked && task.oracle?.geometry && panel === task) {
    group.append(el("path", { d: pathData(task.oracle.geometry, project), class: "route-oracle" }));
  }
  if (toggles.agent.checked && result?.agent_geometry && panel === task) {
    group.append(el("path", { d: pathData(result.agent_geometry, project), class: "route-agent" }));
  } else if (toggles.agent.checked && result?.segment_results && panel.segment_id) {
    const segmentResult = result.segment_results.find((item) => item.task_id.endsWith(`_${panel.segment_id.toLowerCase()}`));
    if (segmentResult?.agent_geometry?.length) {
      group.append(el("path", { d: pathData(segmentResult.agent_geometry, project), class: "route-agent" }));
    }
  }
  return group;
}

function updateMetrics(task) {
  const result = state.results.get(task.task_id);
  const prediction = state.predictions.get(task.task_id);
  const rows = [
    metric("Type", task.task_type || "flat"),
    metric("Distance", `${Math.round(task.oracle?.distance_m || 0)} m`),
  ];
  if (task.task_type === "route_strip") rows.push(metric("Segments", String(task.segments.length)));
  if (result) {
    rows.push(metric("Score", Number(result.score || 0).toFixed(3)));
    rows.push(metric("Valid Route", result.valid_route ? "yes" : "no"));
    rows.push(metric("Length Ratio", Number(result.length_ratio || 0).toFixed(3)));
    rows.push(metric("Mean Distance", `${Number(result.mean_route_distance_m || 0).toFixed(1)} m`));
  }
  if (prediction) rows.push(metric("Prediction", "loaded"));
  metrics.replaceChildren(...rows);
}

function draw() {
  const task = selectedTask();
  if (!task) return;
  updateTaskNav();
  const panel = selectedPanel(task);
  const bbox = panel.task_bbox || task.task_bbox;
  const [west, south, east, north] = bbox;
  const ratio = Math.max(0.55, Math.min(1.8, (east - west) / (north - south || 1)));
  const width = 1000;
  const height = Math.round(width / ratio);
  state.view.width = width;
  state.view.height = height;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const viewport = el("g");
  state.viewport = viewport;
  const project = projectFactory(bbox, width, height);
  const scale = Math.max(0.8, Math.min(1.8, height / 700));
  viewport.append(drawEdges(panel, project, scale));
  viewport.append(drawSegmentBoxes(task, project));
  viewport.append(drawRoutes(task, panel, project));
  viewport.append(drawCheckpoints(task, panel, project, width, height));
  viewport.append(drawMarkers(panel, project));
  applyViewportTransform();
  svg.append(viewport);
  updateMetrics(task);
}

svg.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const point = svgPoint(event);
    const oldK = state.view.k;
    const nextK = Math.max(zoomLimits.min, Math.min(zoomLimits.max, oldK * Math.exp(-event.deltaY * 0.0012)));
    const ratio = nextK / oldK;
    state.view.x = point.x - (point.x - state.view.x) * ratio;
    state.view.y = point.y - (point.y - state.view.y) * ratio;
    state.view.k = nextK;
    applyViewportTransform();
  },
  { passive: false },
);

svg.addEventListener("pointerdown", (event) => {
  svg.setPointerCapture(event.pointerId);
  const rect = svg.getBoundingClientRect();
  state.dragging = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    unitsX: state.view.width / rect.width,
    unitsY: state.view.height / rect.height,
  };
});

svg.addEventListener("pointermove", (event) => {
  const drag = state.dragging;
  if (!drag || drag.pointerId !== event.pointerId) return;
  state.view.x += (event.clientX - drag.x) * drag.unitsX;
  state.view.y += (event.clientY - drag.y) * drag.unitsY;
  drag.x = event.clientX;
  drag.y = event.clientY;
  applyViewportTransform();
});

for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) {
  svg.addEventListener(eventName, (event) => {
    if (state.dragging?.pointerId === event.pointerId) state.dragging = null;
  });
}

async function main() {
  state.tasks = await readJsonl(taskUrl);
  for (const prediction of await readJsonl(predictionUrl, true)) state.predictions.set(prediction.task_id, prediction);
  for (const result of await readJsonl(resultUrl, true)) state.results.set(result.task_id, result);
  subtitle.textContent = `${state.tasks.length} tasks from ${taskUrl}`;
  setupSelectors();
  draw();
}

main().catch((error) => {
  subtitle.textContent = error.message;
  console.error(error);
});
