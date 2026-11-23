const SEQUENTIAL = ["#122241", "#324bb0", "#148dc8", "#4ecd9c", "#ffe370"];
const SIGNED = ["#29a9f3", "#2f68c2", "#18243b", "#b94848", "#ffc568"];
const FOCUS_POSITIVE = "#ffd66b";
const FOCUS_NEGATIVE = "#62ccff";
const compact = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });

function node(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function format(value) {
  if (!Number.isFinite(value)) return "Unmeasured";
  const magnitude = Math.abs(value);
  return magnitude > 0 && (magnitude < .001 || magnitude >= 100000)
    ? value.toExponential(2)
    : compact.format(value);
}

function limits(view) {
  let min = Infinity;
  let max = -Infinity;
  for (const row of view.values) {
    for (const value of row) {
      if (value === null) continue;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  const hasData = Number.isFinite(min);
  const signed = view.signed || min < 0;
  if (!hasData) return { min: 0, max: 0, signed, hasData };
  if (!signed) return { min, max, signed, hasData };
  const bound = Math.max(-min, max);
  return { min: -bound, max: bound, signed, hasData };
}

function interpolate(palette, fraction) {
  const index = Math.max(0, Math.min(1, fraction)) * (palette.length - 1);
  const start = Math.floor(index);
  const end = Math.min(start + 1, palette.length - 1);
  const t = index - start;
  const color = [1, 3, 5].map(offset => {
    const a = parseInt(palette[start].slice(offset, offset + 2), 16);
    const b = parseInt(palette[end].slice(offset, offset + 2), 16);
    return Math.round(a + t * (b - a));
  });
  return `rgb(${color.join(" ")})`;
}

function prepareColors(values, scale) {
  const palette = scale.signed ? SIGNED : SEQUENTIAL;
  const range = scale.max - scale.min;
  return values.map(row => row.map(value => {
    if (value === null) return null;
    let fraction = 0;
    if (scale.signed && scale.max > 0) fraction = .5 + .5 * (value / scale.max);
    else if (range > 0) fraction = (value - scale.min) / range;
    else if (scale.signed || value !== 0) fraction = .5;
    const strength = scale.signed ? Math.abs(2 * fraction - 1) : fraction;
    return { color: interpolate(palette, fraction), strength };
  }));
}

export function signalFocus(values, fraction) {
  const patches = values.flat().flatMap((value, index) => Number.isFinite(value) && value !== 0
    ? [{ index, magnitude: Math.abs(value) }] : []);
  patches.sort((a, b) => b.magnitude - a.magnitude);
  const indices = new Set();
  if (!patches.length) return { indices, peak: null, covered: null, max: 0 };
  const max = patches[0].magnitude;
  const total = patches.reduce((sum, { magnitude }) => sum + magnitude / max, 0);
  let sum = 0;
  let previous;
  for (const { index, magnitude } of patches) {
    if (sum >= fraction * total && magnitude !== previous) break;
    indices.add(index);
    sum += magnitude / max;
    previous = magnitude;
  }
  return {
    indices,
    peak: patches[1]?.magnitude === max ? null : patches[0].index,
    covered: sum / total,
    max,
  };
}

function drawGrid(ctx, w, h, rows, columns) {
  ctx.strokeStyle = "#fff8";
  ctx.lineWidth = Math.min(w, h) / 480;
  ctx.beginPath();
  for (let x = 1; x < columns; x++) {
    ctx.moveTo(x * w / columns, 0);
    ctx.lineTo(x * w / columns, h);
  }
  for (let y = 1; y < rows; y++) {
    ctx.moveTo(0, y * h / rows);
    ctx.lineTo(w, y * h / rows);
  }
  ctx.stroke();
}

function drawRegions(ctx, w, h, image) {
  if (!image?.boxes?.length) return;
  for (const box of image.boxes) {
    const x = box.x * w / image.width;
    const y = box.y * h / image.height;
    const width = box.width * w / image.width;
    const height = box.height * h / image.height;
    const ratio = window.devicePixelRatio || 1;
    ctx.strokeStyle = "#06291f";
    ctx.lineWidth = 4.6 * ratio;
    ctx.strokeRect(x, y, width, height);
    ctx.strokeStyle = "#78ffd4";
    ctx.lineWidth = 2.2 * ratio;
    ctx.strokeRect(x, y, width, height);
  }
}

function drawTargetBoundary(ctx, w, h, rows, columns, patches) {
  const dx = w / columns;
  const dy = h / rows;
  const ratio = window.devicePixelRatio || 1;
  ctx.save();
  ctx.beginPath();
  function edge(x1, y1, x2, y2) {
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  for (const index of patches) {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const x = col * dx;
    const y = row * dy;
    if (row === 0 || !patches.has(index - columns)) edge(x, y, x + dx, y);
    if (col === 0 || !patches.has(index - 1)) edge(x, y, x, y + dy);
    if (col === columns - 1 || !patches.has(index + 1)) edge(x + dx, y, x + dx, y + dy);
    if (row === rows - 1 || !patches.has(index + columns)) edge(x, y + dy, x + dx, y + dy);
  }
  ctx.setLineDash([3 * ratio, 2 * ratio]);
  ctx.strokeStyle = "#75ffd3";
  ctx.lineWidth = 1.25 * ratio;
  ctx.stroke();
  ctx.restore();
}

function drawSelected(ctx, patch, w, h, rows, cols) {
  if (!patch) return;
  const x = patch.col * w / cols;
  const y = patch.row * h / rows;
  const ratio = window.devicePixelRatio || 1;
  ctx.strokeStyle = "#0d192b";
  ctx.lineWidth = ratio * 4;
  ctx.strokeRect(x, y, w / cols, h / rows);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = ratio * 1.8;
  ctx.strokeRect(x, y, w / cols, h / rows);
}

function drawPeak(ctx, w, h, rows, columns, focus, values) {
  if (focus.peak === null) return;
  const x = focus.peak % columns;
  const y = Math.floor(focus.peak / columns);
  const color = values[y][x] < 0 ? FOCUS_NEGATIVE : FOCUS_POSITIVE;
  ctx.beginPath();
  ctx.arc((x + .5) * w / columns, (y + .5) * h / rows,
    .63 * Math.min(w / columns, h / rows), 0, 2 * Math.PI);
  ctx.strokeStyle = "#152435";
  ctx.lineWidth = 4.5 * (window.devicePixelRatio || 1);
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.stroke();
}

function drawMap(ctx, w, h, { colors, values, image, options, reference, targetPatches, mode, focus, patch }) {
  const rows = colors.length;
  const columns = colors[0].length;
  ctx.clearRect(0, 0, w, h);
  if (reference && options.grid && targetPatches) {
    ctx.fillStyle = "#6cffcb55";
    for (const cell of targetPatches) {
      const x = cell % columns;
      const y = Math.floor(cell / columns);
      ctx.fillRect(x * w / columns, y * h / rows, w / columns, h / rows);
    }
  }
  if (mode !== "image" && !reference) {
    colors.forEach((row, y) => row.forEach((cell, x) => {
      const left = Math.round(x * w / columns);
      const top = Math.round(y * h / rows);
      const width = Math.round((x + 1) * w / columns) - left;
      const height = Math.round((y + 1) * h / rows) - top;
      if (mode === "focus") {
        if (!focus.indices.has(y * columns + x)) return;
        const value = values[y][x];
        const color = value < 0 ? FOCUS_NEGATIVE : FOCUS_POSITIVE;
        ctx.globalAlpha = .17 + .7 * Math.pow(Math.abs(value) / focus.max, .42);
        ctx.fillStyle = color;
        ctx.fillRect(left, top, width, height);
        return;
      }
      if (!cell) {
        if (mode === "heatmap") {
          ctx.fillStyle = (x + y) % 2 ? "#788599" : "#a5afbd";
          ctx.fillRect(left, top, width, height);
        }
        return;
      }
      ctx.globalAlpha = mode === "heatmap" ? 1 : options.opacity * Math.pow(cell.strength, .33);
      ctx.fillStyle = cell.color;
      ctx.fillRect(left, top, width, height);
    }));
  }
  ctx.globalAlpha = 1;
  if (options.grid) drawGrid(ctx, w, h, rows, columns);
  if (mode === "focus" && options.boxes && targetPatches) {
    drawTargetBoundary(ctx, w, h, rows, columns, targetPatches);
  }
  if (reference || options.boxes) drawRegions(ctx, w, h, image);
  if (mode === "focus") drawPeak(ctx, w, h, rows, columns, focus, values);
  drawSelected(ctx, patch, w, h, rows, columns);
}

function addImage(media, image) {
  if (!image) return;
  const photo = node("img", "map-image");
  photo.alt = "";
  photo.src = image.src;
  photo.draggable = false;
  const frame = image.content || { x: 0, y: 0, width: image.width, height: image.height };
  photo.style.left = `${100 * frame.x / image.width}%`;
  photo.style.top = `${100 * frame.y / image.height}%`;
  photo.style.width = `${100 * frame.width / image.width}%`;
  photo.style.height = `${100 * frame.height / image.height}%`;
  media.append(photo);
}

function legendFor(scale, view) {
  const legend = node("div", "map-legend");
  if (!scale.hasData) return legend;
  const strip = node("div", "map-scale");
  const palette = scale.signed ? SIGNED : SEQUENTIAL;
  strip.style.background = `linear-gradient(to right, ${palette.join(", ")})`;
  const ticks = node("div", "map-ticks");
  const values = scale.signed ? [scale.min, 0, scale.max] : [scale.min, scale.max];
  for (const tick of new Set(values)) ticks.append(node("span", "", format(tick)));
  legend.append(strip, ticks);
  legend.setAttribute("aria-label", `${view.unit || "Value"} · ${format(scale.min)} to ${format(scale.max)}`);
  return legend;
}

export function spatialKey(view, image) {
  return JSON.stringify([image?.src, image?.width, image?.height, image?.content, view.values.length, view.values[0].length]);
}

export function renderSpatial(view, image, {
  options, updatePatch, currentPatch, readoutId, reference = false, targetPatches,
}) {
  const host = node("figure", "map-figure");
  const media = node("div", "map-media");
  const canvas = node("canvas", "map-canvas");
  const chip = node("span", "map-overlay-readout");
  const readout = node("figcaption", "map-readout");
  const rows = view.values.length;
  const cols = view.values[0].length;
  const scale = limits(view);
  const colors = reference ? view.values : prepareColors(view.values, scale);
  const key = spatialKey(view, image);
  const width = image?.width || cols;
  const height = image?.height || rows;
  let patch = currentPatch(key);
  let focusFraction;
  let focus;
  const paint = { colors, values: view.values, image, options, reference, targetPatches };

  media.style.aspectRatio = `${width} / ${height}`;
  media.style.maxWidth = `${570 * width / height}px`;
  media.style.backgroundColor = image?.background || "#132139";
  media.tabIndex = 0;
  const title = reference ? (image.boxes?.length ? "Reference image with dataset annotation" : "Reference image") : view.title;
  media.setAttribute("aria-label", `${title}. Use arrow keys to inspect patches, Enter to pin, Escape to clear.`);
  media.setAttribute("aria-describedby", readoutId);
  canvas.setAttribute("aria-hidden", "true");
  chip.setAttribute("aria-hidden", "true");
  readout.id = readoutId;
  addImage(media, image);
  media.append(canvas, chip);
  if (reference) {
    image.boxes?.filter(box => box.label).forEach(box => {
      const label = node("span", "reference-box-label", box.label);
      label.style.left = `${Math.min(70, 100 * box.x / image.width)}%`;
      label.style.top = `${100 * box.y / image.height}%`;
      media.append(label);
    });
  }
  const legend = legendFor(scale, view);
  host.append(media);
  if (!reference) host.append(legend);
  host.append(readout);

  const ctx = canvas.getContext("2d");
  function refresh() {
    const mode = reference ? "image" : image ? options.mode : "heatmap";
    media.dataset.mode = mode;
    legend.hidden = mode === "focus" || mode === "image";
    if (mode === "focus" && focusFraction !== options.focusFraction) {
      focusFraction = options.focusFraction;
      focus = signalFocus(view.values, focusFraction);
    }
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    const w = Math.round(media.clientWidth * ratio);
    const h = Math.round(media.clientHeight * ratio);
    if (ctx && w && h) {
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      drawMap(ctx, w, h, { ...paint, mode, focus, patch });
    }
    chip.hidden = !patch;
    if (patch) {
      const raw = view.values[patch.row][patch.col];
      const position = `Row ${patch.row + 1}, col ${patch.col + 1}`;
      const inBox = targetPatches?.has(patch.row * cols + patch.col);
      const region = targetPatches ? (inBox ? " · Target patch" : " · Outside target patches") : "";
      chip.textContent = reference ? position : `${position} · ${format(raw)}`;
      const value = reference ? "" : ` · ${raw ?? "Unmeasured"}`;
      readout.textContent = `${patch.locked ? "Pinned · " : ""}${position}${value}${region}`;
    } else if (reference) {
      readout.textContent = image.boxes?.length ? "Mint outline · dataset annotation" : `${rows} × ${cols} model patches`;
    } else if (mode === "focus") {
      const count = focus.indices.size;
      readout.textContent = count
        ? `${count} highlighted ${count === 1 ? "patch" : "patches"} · ${(100 * focus.covered).toFixed(1)}% of signal`
        : scale.hasData ? "No signal in this map" : "No measured patches";
    } else {
      readout.textContent = `${rows} × ${cols} patches · ${scale.hasData ? "Inspect a patch to see its value" : "No measured patches"}`;
    }
  }

  function toPatch(event) {
    const rect = media.getBoundingClientRect();
    return {
      col: Math.min(cols - 1, Math.max(0, Math.floor((event.clientX - rect.left) / rect.width * cols))),
      row: Math.min(rows - 1, Math.max(0, Math.floor((event.clientY - rect.top) / rect.height * rows))),
    };
  }

  media.addEventListener("pointermove", event => {
    if (event.pointerType !== "touch" && !patch?.locked) updatePatch(key, toPatch(event));
  });
  media.addEventListener("pointerleave", () => {
    if (!patch?.locked && document.activeElement !== media) updatePatch(key, null);
  });
  media.addEventListener("click", event => {
    const next = toPatch(event);
    const same = patch?.locked && next.row === patch.row && next.col === patch.col;
    updatePatch(key, same ? null : { ...next, locked: true });
  });
  media.addEventListener("focus", () => {
    if (!patch) updatePatch(key, { row: 0, col: 0 });
  });
  media.addEventListener("blur", () => {
    if (!patch?.locked) updatePatch(key, null);
  });
  media.addEventListener("keydown", event => {
    const delta = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
    if (delta) {
      event.preventDefault();
      updatePatch(key, {
        row: Math.max(0, Math.min(rows - 1, (patch?.row || 0) + delta[0])),
        col: Math.max(0, Math.min(cols - 1, (patch?.col || 0) + delta[1])),
        locked: patch?.locked,
      });
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      updatePatch(key, { row: patch?.row || 0, col: patch?.col || 0, locked: !patch?.locked });
    } else if (event.key === "Escape") {
      updatePatch(key, null);
    }
  });

  const observer = new ResizeObserver(refresh);
  observer.observe(media);
  return {
    host,
    key,
    refresh,
    select(next) {
      if (patch?.row === next?.row && patch?.col === next?.col && patch?.locked === next?.locked) return;
      patch = next;
      refresh();
    },
    destroy() { observer.disconnect(); },
  };
}
