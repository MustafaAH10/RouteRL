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
const panelSelect = document.querySelector("#panelSelect");
const panelField = document.querySelector("#panelField");
const subtitle = document.querySelector("#subtitle");
const metrics = document.querySelector("#metrics");
const toggles = {
  graph: document.querySelector("#graphToggle"),
  oracle: document.querySelector("#oracleToggle"),
  agent: document.querySelector("#agentToggle"),
  checkpoints: document.querySelector("#checkpointsToggle"),
};

const state = {
  tasks: [],
  predictions: new Map(),
  results: new Map(),
};

async function readJsonl(url, optional = false) {
  if (!url) return [];
  const response = await fetch(url);
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
  return state.tasks[Number(taskSelect.value) || 0];
}

function selectedPanel(task) {
  if (task.task_type !== "route_strip") return task;
  const value = panelSelect.value || "overview";
  if (value === "overview") return task;
  return task.segments.find((segment) => segment.segment_id === value) || task;
}

function setupSelectors() {
  taskSelect.replaceChildren(
    ...state.tasks.map((task, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = task.task_id;
      return option;
    }),
  );
  taskSelect.addEventListener("change", () => {
    setupPanelSelector();
    draw();
  });
  panelSelect.addEventListener("change", draw);
  for (const toggle of Object.values(toggles)) toggle.addEventListener("change", draw);
  setupPanelSelector();
}

function setupPanelSelector() {
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
  }
  return group;
}

function drawCheckpoints(panel, project) {
  const group = el("g");
  if (!toggles.checkpoints.checked) return group;
  for (const [label, point] of Object.entries(panel.turn_checkpoints || {})) {
    const [x, y] = project([point.lon, point.lat]);
    group.append(el("circle", { cx: x, cy: y, r: 4.2, fill: "#111", stroke: "#fff", "stroke-width": 1.3 }));
    group.append(el("text", { x: x + 6, y: y - 7, class: "checkpoint-label" }, label));
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
  const panel = selectedPanel(task);
  const bbox = panel.task_bbox || task.task_bbox;
  const [west, south, east, north] = bbox;
  const ratio = Math.max(0.55, Math.min(1.8, (east - west) / (north - south || 1)));
  const width = 1000;
  const height = Math.round(width / ratio);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const project = projectFactory(bbox, width, height);
  const scale = Math.max(0.8, Math.min(1.8, height / 700));
  svg.append(drawEdges(panel, project, scale));
  svg.append(drawSegmentBoxes(task, project));
  svg.append(drawRoutes(task, panel, project));
  svg.append(drawCheckpoints(panel, project));
  svg.append(drawMarkers(panel, project));
  updateMetrics(task);
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
