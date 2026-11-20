const compactNumber = new Intl.NumberFormat("en", { maximumSignificantDigits: 5 });
const traceColors = ["#1e7063", "#b94f3c", "#365ba4", "#794ea1", "#9a6609", "#3d7790"];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function format(value) {
  if (value === null || value === undefined) return "Unmeasured";
  if (typeof value !== "number") return String(value);
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude > 0 && (magnitude < 0.0001 || magnitude >= 1e6)) {
    return value.toExponential(3);
  }
  return compactNumber.format(value);
}

function measurement(value) {
  return Number.isFinite(value) ? String(value) : "Unmeasured";
}

function chartDomain(rows, { signed, lowerIsBetter, bounds }) {
  let min = Infinity;
  let max = -Infinity;
  for (const values of rows) {
    for (const value of values) {
      if (!Number.isFinite(value) || (lowerIsBetter && value < 1)) continue;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  if (min === Infinity) return null;
  if (bounds?.length === 2) {
    const [lower, upper] = bounds;
    if (Number.isFinite(lower) && Number.isFinite(upper) && lower <= min && upper >= max) {
      min = lower;
      max = upper;
    }
  }
  if (lowerIsBetter) return [1, max];
  if (signed) {
    const limit = Math.max(Math.abs(min), Math.abs(max));
    return [-limit, limit];
  }
  return [min, max];
}

function dataTable(columns, rows, caption, exact = false) {
  const scroll = element("div", "chart-scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", caption);
  const table = element("table", "chart-table");
  const header = table.createTHead().insertRow();
  for (const label of columns) {
    const cell = element("th", "", label);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) {
      const cell = row.insertCell();
      cell.textContent = exact && typeof value === "number" ? String(value) : format(value);
      if (typeof value === "number") cell.title = String(value);
    }
  }
  scroll.append(table);
  return scroll;
}

function heatColor(value, domain, view) {
  const [min, max] = domain;
  const neutral = "var(--chart-neutral, #f0efe8)";
  const positive = "var(--chart-positive, #1e7063)";
  const negative = "var(--chart-negative, #b94f3c)";
  let color = positive;
  let magnitude;
  if (view.lowerIsBetter) {
    magnitude = max === 1 ? 1 : 1 - Math.log(value) / Math.log(max);
  } else if (view.signed) {
    color = value < 0 ? negative : positive;
    magnitude = max === 0 ? 0 : Math.abs(value) / max;
  } else {
    magnitude = max === min ? 0 : (value - min) / (max - min);
  }
  const percent = 100 * Math.max(0, Math.min(1, magnitude));
  return `color-mix(in srgb, ${color} ${percent}%, ${neutral})`;
}

function matrixLegend(domain, view, hasMissing) {
  const legend = element("div", "chart-legend");
  if (domain) {
    const [min, max] = domain;
    const scale = element("div", "chart-scale");
    const stops = view.signed && !view.lowerIsBetter ? [min, 0, max] : [min, max];
    const colors = stops.map(value => heatColor(value, domain, view));
    scale.style.background = `linear-gradient(to right, ${colors.join(", ")})`;
    scale.setAttribute("aria-hidden", "true");
    const labels = element("div", "chart-scale-labels");
    for (const value of new Set(stops)) labels.append(element("span", "", format(value)));
    legend.append(scale, labels);
  }
  const note = [view.lowerIsBetter ? "Rank · logarithmic color scale" : view.unit,
    hasMissing ? "Hatched cells are unmeasured" : null].filter(Boolean).join(" · ");
  if (note) legend.append(element("div", "chart-legend-note", note));
  return legend;
}

function renderMatrix(view, host) {
  const domain = chartDomain(view.values, view);
  const scroll = element("div", "chart-scroll is-matrix");
  const table = element("table", "chart-matrix");
  table.setAttribute("aria-label", `${view.title}. Use arrow keys to inspect cells.`);
  const header = table.createTHead().insertRow();
  header.append(element("th", "", view.rowLabel || ""));
  for (const label of view.columns) {
    const cell = element("th", "", label);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  const readout = element("div", "chart-readout");
  const buttons = [];
  let firstMeasured;
  let selected;
  let hasMissing = false;

  function select(button) {
    if (selected) selected.tabIndex = -1;
    selected = button;
    selected.tabIndex = 0;
    inspect(button);
  }

  function inspect(button) {
    readout.textContent = button.getAttribute("aria-label");
  }

  view.rows.forEach((label, rowIndex) => {
    const row = body.insertRow();
    const heading = element("th", "", label);
    heading.scope = "row";
    row.append(heading);
    const rowButtons = view.columns.map((column, columnIndex) => {
      const value = view.values[rowIndex][columnIndex];
      const measured = Number.isFinite(value) && (!view.lowerIsBetter || value >= 1);
      const button = element("button", measured ? "chart-cell" : "chart-cell is-missing");
      button.type = "button";
      button.tabIndex = -1;
      button.dataset.row = rowIndex;
      button.dataset.column = columnIndex;
      const rowText = view.rowLabel ? `${view.rowLabel} ${label}` : label;
      const columnText = view.columnLabel ? `${view.columnLabel} ${column}` : column;
      const unit = measured && view.unit ? ` ${view.unit}` : "";
      button.setAttribute("aria-label", `${rowText} · ${columnText} · ${measurement(measured ? value : null)}${unit}`);
      if (measured) {
        button.style.setProperty("--chart-cell-color", heatColor(value, domain, view));
        firstMeasured ??= button;
      } else {
        hasMissing = true;
      }
      button.addEventListener("pointerenter", () => inspect(button));
      button.addEventListener("focus", () => select(button));
      button.addEventListener("click", () => select(button));
      row.insertCell().append(button);
      return button;
    });
    buttons.push(rowButtons);
  });
  const first = firstMeasured || buttons[0]?.[0];
  if (first) select(first);
  table.addEventListener("pointerleave", () => selected && inspect(selected));
  const directions = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  table.addEventListener("keydown", event => {
    const direction = directions[event.key];
    if (!direction || !event.target.matches(".chart-cell")) return;
    event.preventDefault();
    const row = Number(event.target.dataset.row) + direction[0];
    const column = Number(event.target.dataset.column) + direction[1];
    buttons[row]?.[column]?.focus();
  });
  scroll.append(table);
  host.append(scroll, matrixLegend(domain, view, hasMissing), readout);
}

function renderBars(view, host) {
  const rows = view.entries.map(entry => [entry.value]);
  const signed = view.signed || view.entries.some(entry => Number.isFinite(entry.value) && entry.value < 0);
  const limit = chartDomain(rows, { signed, bounds: view.bounds })?.[1] || 1;
  const bars = element("div", "chart-bars");
  for (const { label, value } of view.entries) {
    const row = element("div", "chart-bar-row");
    const track = element("div", signed ? "chart-bar-track is-signed" : "chart-bar-track");
    track.setAttribute("aria-hidden", "true");
    const score = element("span", "chart-bar-value", format(value));
    if (Number.isFinite(value)) {
      const share = Math.abs(value) / limit * (signed ? 50 : 100);
      const fill = element("div", value < 0 ? "chart-bar-fill is-negative" : "chart-bar-fill");
      fill.style.inlineSize = `${share}%`;
      fill.style.insetInlineStart = `${signed ? 50 - (value < 0 ? share : 0) : 0}%`;
      track.append(fill);
      score.title = String(value);
    } else {
      track.classList.add("is-missing");
    }
    row.append(element("span", "chart-bar-label", label), track, score);
    bars.append(row);
  }
  host.append(bars);
  if (view.unit) host.append(element("div", "chart-legend-note", view.unit));
}

function lineXPositions(values) {
  if (!values.every(Number.isFinite)) return values.map((_, index) => index / (values.length - 1 || 1));
  const [min, max] = chartDomain([values], {}) || [0, 0];
  return values.map(value => max === min ? 0.5 : (value - min) / (max - min));
}

function shorten(text, length) {
  text = String(text);
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function drawSeries(canvas, view, state) {
  const bounds = canvas.getBoundingClientRect();
  const width = bounds.width;
  const height = bounds.height;
  if (width < 1 || height < 1) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * pixelRatio);
  const pixelHeight = Math.round(height * pixelRatio);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

  const context = canvas.getContext("2d");
  if (!context) return;
  const style = getComputedStyle(canvas);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.font = "11px system-ui, sans-serif";
  const ink = style.getPropertyValue("--chart-ink").trim() || "#51554d";
  const grid = style.getPropertyValue("--chart-grid").trim() || "#dfdfd6";
  let [min, max] = state.domain;
  if (min === max && !view.lowerIsBetter) {
    const pad = Math.abs(min) * 0.1 || 1;
    min -= pad;
    max += pad;
  }
  const middle = min / 2 + max / 2;
  const ticks = view.lowerIsBetter ? [...new Set([min, Math.round(Math.sqrt(max)), max])] : [min, middle, max];
  const labelWidth = Math.max(...ticks.map(value => context.measureText(format(value)).width));
  const plot = { left: labelWidth + 12, top: 13, right: width - 15, bottom: height - 26 };
  if (plot.right <= plot.left || plot.bottom <= plot.top) return;
  const x = index => plot.left + (plot.right - plot.left) * state.positions[index];
  const y = value => {
    if (min === max) return (plot.top + plot.bottom) / 2;
    const ratio = view.lowerIsBetter ? Math.log(value) / Math.log(max) : (max - value) / (max - min);
    return plot.top + (plot.bottom - plot.top) * ratio;
  };
  state.xPixels = state.positions.map((_, index) => x(index));
  context.fillStyle = ink;
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (const tick of ticks) {
    const lineY = y(tick);
    context.strokeStyle = tick === 0 && view.signed ? ink : grid;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(plot.left, lineY);
    context.lineTo(plot.right, lineY);
    context.stroke();
    context.fillText(format(tick), plot.left - 7, lineY);
  }

  const labels = Math.min(view.x.length, Math.max(2, Math.floor((plot.right - plot.left) / 65)));
  context.textBaseline = "top";
  for (let index = 0; index < labels; index += 1) {
    const point = Math.round(index * (view.x.length - 1) / (labels - 1 || 1));
    const text = shorten(view.x[point], 10);
    const labelWidth = context.measureText(text).width;
    const left = Math.max(0, Math.min(width - labelWidth, x(point) - labelWidth / 2));
    context.textAlign = "left";
    context.fillText(text, left, plot.bottom + 9);
  }

  if (state.active !== null) {
    context.strokeStyle = ink;
    context.lineWidth = 1;
    context.setLineDash([3, 4]);
    context.beginPath();
    context.moveTo(x(state.active), plot.top);
    context.lineTo(x(state.active), plot.bottom);
    context.stroke();
    context.setLineDash([]);
  }
  view.series.forEach(({ values }, seriesIndex) => {
    const color = traceColors[seriesIndex % traceColors.length];
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.beginPath();
    let connected = false;
    values.forEach((value, valueIndex) => {
      if (!Number.isFinite(value) || (view.lowerIsBetter && value < 1)) {
        connected = false;
        return;
      }
      if (connected) context.lineTo(x(valueIndex), y(value));
      else context.moveTo(x(valueIndex), y(value));
      connected = true;
    });
    context.stroke();
    values.forEach((value, valueIndex) => {
      if (!Number.isFinite(value) || (view.lowerIsBetter && value < 1)) return;
      const active = valueIndex === state.active;
      const isolated = !Number.isFinite(values[valueIndex - 1]) && !Number.isFinite(values[valueIndex + 1]);
      if (!active && !isolated && values.length > 14) return;
      context.beginPath();
      context.arc(x(valueIndex), y(value), active ? 4 : 2.2, 0, 2 * Math.PI);
      context.fill();
    });
  });
}

function renderSeries(view, host) {
  const domain = chartDomain(view.series.map(trace => trace.values), view);
  if (!domain || !view.x.length) {
    host.append(element("p", "chart-readout", "No measured values in this view."));
    return;
  }
  const state = { domain, positions: lineXPositions(view.x), active: null, xPixels: [] };
  const canvas = element("canvas", "chart-lines");
  canvas.style.display = "block";
  canvas.style.width = "100%";
  canvas.style.aspectRatio = "1.8";
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "slider");
  canvas.setAttribute("aria-label", `${view.title}. Use arrow keys to inspect data points.`);
  canvas.setAttribute("aria-valuemin", "1");
  canvas.setAttribute("aria-valuemax", String(view.x.length));
  const key = element("ul", "chart-line-key");
  view.series.forEach(({ label }, index) => {
    const item = element("li");
    const swatch = element("span", "chart-line-swatch");
    swatch.style.backgroundColor = traceColors[index % traceColors.length];
    swatch.setAttribute("aria-hidden", "true");
    item.append(swatch, element("span", "", label));
    key.append(item);
  });
  const readout = element("div", "chart-readout");
  const xLabel = view.xLabel || "Position";
  function inspect(index) {
    const xText = `${xLabel} ${view.x[index]}`;
    const scores = view.series.map(({ label, values }) => `${label} ${measurement(values[index])}`);
    const summary = [xText, ...scores].join(" · ");
    canvas.setAttribute("aria-valuenow", String(index + 1));
    canvas.setAttribute("aria-valuetext", summary);
    readout.textContent = summary;
    if (state.active !== index) {
      state.active = index;
      drawSeries(canvas, view, state);
    }
  }

  const note = [view.unit, view.lowerIsBetter ? "Logarithmic rank axis" : null]
    .filter(Boolean).join(" · ");
  const details = element("details", "chart-data");
  details.append(element("summary", "", "Measured values"));
  details.addEventListener("toggle", () => {
    if (!details.open || details.childElementCount > 1) return;
    const labels = [xLabel, ...view.series.map(trace => trace.label)];
    const rows = view.x.map((x, index) => [x, ...view.series.map(trace => trace.values[index])]);
    details.append(dataTable(labels, rows, view.title, true));
  });

  canvas.addEventListener("pointermove", event => {
    const point = event.clientX - canvas.getBoundingClientRect().left;
    let nearest = 0;
    state.xPixels.forEach((pixel, index) => {
      if (Math.abs(pixel - point) < Math.abs(state.xPixels[nearest] - point)) nearest = index;
    });
    inspect(nearest);
  });
  canvas.addEventListener("keydown", event => {
    const moves = {
      ArrowRight: state.active + 1, ArrowUp: state.active + 1,
      ArrowLeft: state.active - 1, ArrowDown: state.active - 1,
      Home: 0, End: view.x.length - 1,
    };
    if (!Object.hasOwn(moves, event.key)) return;
    event.preventDefault();
    inspect(Math.min(view.x.length - 1, Math.max(0, moves[event.key])));
  });
  host.append(canvas, key, readout, details);
  if (note) host.insertBefore(element("div", "chart-legend-note", note), readout);
  inspect(0);

  const observer = new ResizeObserver(() => {
    if (canvas.isConnected) drawSeries(canvas, view, state);
    else observer.disconnect();
  });
  observer.observe(canvas);
}

export function renderChart(view) {
  const host = element("div", "chart");
  if (view.type === "matrix") renderMatrix(view, host);
  if (view.type === "bars") renderBars(view, host);
  if (view.type === "series") renderSeries(view, host);
  if (view.type === "table") host.append(dataTable(view.columns, view.rows, view.title));
  return host;
}
