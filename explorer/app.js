import { renderChart } from "./charts.js";
import { validateBundle, sourceHref, families } from "./bundle.js";
import { renderSpatial, spatialKey } from "./spatial.js";
import { renderEvidence } from "./evidence.js";

const byId = id => document.getElementById(id);
const repository = document.querySelector(".header-link").href;
const colors = ["#365bca", "#1e8270", "#b66047", "#754fc0", "#963e78", "#35798d"];
const methodLimit = 4;
const referenceKey = Symbol("reference");
const percent = new Intl.NumberFormat("en", {
  style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1,
});
const ui = {
  families: byId("families"), caseSelect: byId("case-select"), summary: byId("case-summary"),
  description: byId("family-description"), comparison: byId("comparison"), comparisonTitle: byId("comparison-title"),
  methods: byId("methods"), count: byId("compare-count"), panels: byId("panels"), notice: byId("notice"),
  source: byId("case-source"), toolbar: byId("map-toolbar"), modes: byId("map-modes"), scaleNote: byId("map-scale-note"),
  localization: byId("localization"), evidence: byId("evidence"), focusField: byId("focus-field"), focus: byId("focus-fraction"),
  referenceDialog: byId("reference-dialog"), original: byId("reference-original"),
  layer: byId("layer-select"), layerField: byId("layer-field"), opacity: byId("opacity"),
  opacityField: byId("opacity-field"), opacityValue: byId("opacity-value"), grid: byId("show-grid"),
  boxes: byId("show-boxes"), boxesField: byId("boxes-field"), file: byId("import-input"),
  importButton: byId("import-button"),
};
const state = {
  cases: [],
  case: null,
  family: "attention",
  active: new Set(),
  views: new Map(),
  maps: new Map(),
  patches: new Map(),
  evidenceMetric: null,
  options: {
    mode: "focus", focusFraction: Number(ui.focus.value), opacity: Number(ui.opacity.value) / 100,
    grid: ui.grid.checked, boxes: ui.boxes.checked,
  },
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function notice(text, error = false) {
  ui.notice.textContent = text;
  ui.notice.classList.toggle("is-error", error);
  ui.notice.hidden = !text;
}

function renderFamilies() {
  for (const [id, family] of Object.entries(families)) {
    const button = node("button", "family");
    button.type = "button";
    button.dataset.family = id;
    button.style.setProperty("--family-accent", family.color);
    const icon = node("span", "family-icon", family.icon);
    icon.setAttribute("aria-hidden", "true");
    button.append(icon, family.label);
    button.addEventListener("click", () => {
      if (id === state.family) return;
      const next = state.cases.find(c => c.family === id);
      notice("");
      selectCase(next || null, id);
    });
    ui.families.append(button);
  }
}

function renderFamilySelection() {
  for (const button of ui.families.children) {
    button.setAttribute("aria-pressed", String(button.dataset.family === state.family));
  }
  ui.description.textContent = families[state.family].description;
  ui.caseSelect.replaceChildren();
  const cases = state.cases.filter(c => c.family === state.family);
  ui.caseSelect.disabled = cases.length === 0;
  if (!cases.length) ui.caseSelect.add(new Option("No experiments loaded", ""));
  for (const c of cases) {
    ui.caseSelect.add(new Option(c.title, c.key, false, c.key === state.case?.key));
  }
}

function renderSummary(c) {
  const preview = node("img", "sample-media");
  preview.hidden = !c.image;
  if (c.image) {
    preview.src = c.image.src;
    preview.alt = c.image.alt || "Experiment input";
  }
  const input = node("div", "sample-input");
  const kind = c.imported ? "Imported result" : "Archived measurement";
  const label = c.dataset ? `${c.dataset} · ${kind}` : kind;
  input.append(node("p", "sample-label", label), node("p", "sample-prompt", c.prompt || c.title));
  if (c.target) {
    const target = node("p", "sample-answer");
    target.append(node("span", "", state.family === "sae" ? "Feature source" : "Target"), node("strong", "", c.target));
    input.append(target);
  }
  const meta = node("dl", "case-meta");
  for (const [term, value] of [["Model", c.model], ["Measurement", c.metric]]) {
    if (!value) continue;
    const row = node("div");
    row.append(node("dt", "", term), node("dd", "", value));
    meta.append(row);
  }
  meta.hidden = !meta.children.length;
  ui.summary.replaceChildren(preview, input, meta);
  ui.summary.hidden = false;
}

function renderSource(c) {
  const details = node("details");
  details.append(node("summary", "", "Experiment settings and source"));
  if (c.description) details.append(node("p", "", c.description));
  if (c.source?.description) details.append(node("p", "", c.source.description));
  const href = sourceHref(c.source, repository, !c.imported);
  if (href) {
    const link = node("a", "", c.imported ? "Result format documentation" : "Open source measurement on GitHub");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener";
    details.append(link);
  }
  ui.source.replaceChildren(details);
}

function renderMethodSelection() {
  ui.methods.replaceChildren();
  ui.methods.hidden = state.case.methods.length === 1;
  state.case.methods.forEach((method, index) => {
    const chip = node("button", "method-chip");
    chip.type = "button";
    chip.dataset.method = method.id;
    chip.style.setProperty("--chip", colors[index % colors.length]);
    const check = node("span", "method-check", "✓");
    check.setAttribute("aria-hidden", "true");
    chip.append(check, method.label);
    chip.addEventListener("click", () => {
      if (chip.getAttribute("aria-disabled") === "true") return;
      if (state.active.has(method.id)) state.active.delete(method.id);
      else state.active.add(method.id);
      updateChips();
      renderPanels();
    });
    ui.methods.append(chip);
  });
  updateChips();
}

function updateChips() {
  const max = Math.min(state.case.methods.length, methodLimit);
  const selected = state.active.size;
  ui.comparisonTitle.textContent = max === 1 ? "Method result" : "Compare methods";
  ui.count.textContent = max > 1 ? `${selected} of ${max} panels` : "";
  for (const button of ui.methods.children) {
    const active = state.active.has(button.dataset.method);
    const disabled = active ? selected === 1 : selected >= methodLimit;
    button.setAttribute("aria-pressed", String(active));
    button.setAttribute("aria-disabled", String(disabled));
    button.title = disabled
      ? active ? "Keep at least one result visible" : "Deselect a method to free a panel"
      : button.textContent.slice(1);
  }
}

function activeMethods() {
  return state.case.methods.filter(method => state.active.has(method.id));
}

function currentView(method) {
  return method.views[state.views.get(method.id) || 0];
}

function localizationComparison(maps) {
  const measured = maps.filter(map => map.view.localization);
  const signaled = measured.filter(({ view }) => !view.localization.zeroMap);
  const first = measured[0];
  const leaders = new Set();
  if (!first) return null;
  const { areaFraction, region } = first.view.localization;
  const { layer } = first.view;
  const frame = spatialKey(first.view, first.image);
  const matched = measured.every(({ view, image }) => view.layer === layer
    && spatialKey(view, image) === frame && view.localization.areaFraction === areaFraction
    && view.localization.region === region);
  const highest = signaled.reduce((best, { view }) => Math.max(best, view.localization.massFraction), -Infinity);
  if (matched && signaled.length > 1) {
    for (const { method, view } of signaled) {
      if (percent.format(view.localization.massFraction) === percent.format(highest)) leaders.add(method.id);
    }
  }
  return { areaFraction, region, layer, measured, matched, highest, leaders };
}

function renderLocalization(result) {
  ui.localization.replaceChildren();
  ui.localization.hidden = !result;
  if (!result) return;
  const { matched, measured, leaders, highest, layer, areaFraction, region } = result;
  const claim = node("div", "localization-claim");
  const label = node("span", "localization-label", "Share of absolute signal on patches that touch the annotated box");
  let title = "Compare the in-box share on each map";
  if (leaders.size) {
    const names = measured.filter(({ method }) => leaders.has(method.id)).map(({ method }) => method.label);
    title = names.length === 1
      ? `${names[0]} has the highest share at the ${region}`
      : `${names.join(" and ")} tie at displayed precision`;
  } else if (!matched) {
    title = "Select a common layer to compare target shares";
  } else if (!Number.isFinite(highest)) {
    title = "The selected maps have no signal";
  }
  claim.append(label, node("h3", "", title));
  if (leaders.size) {
    const value = node("strong", "localization-leading-value", percent.format(highest));
    claim.append(value);
  }
  const baseline = node("div", "localization-baseline");
  if (matched) {
    baseline.append(node("strong", "", `${percent.format(areaFraction)} uniform baseline`));
    const scope = Number.isFinite(layer) ? `Through L${layer} · ` : "";
    baseline.append(node("span", "", `${scope}${measured.length} selected ${measured.length === 1 ? "map" : "maps"}`));
    if (Number.isFinite(highest) && highest < areaFraction) {
      baseline.append(node("span", "localization-below", "All below baseline"));
    }
  }
  ui.localization.append(claim, baseline);
}

function shareDisplay(value, label, baseline) {
  const display = node("div", "map-share");
  const content = node("div", "map-share-content");
  content.append(node("strong", "map-share-value", Number.isFinite(value) ? percent.format(value) : "No signal"),
    node("span", "map-share-label", label));
  display.append(content);
  if (Number.isFinite(value)) {
    const track = node("div", "map-share-track");
    const fill = node("span", "map-share-fill");
    fill.style.width = `${100 * value}%`;
    track.append(fill);
    if (Number.isFinite(baseline)) {
      const marker = node("span", "map-share-baseline");
      marker.style.left = `${100 * baseline}%`;
      marker.title = `Uniform baseline · ${percent.format(baseline)}`;
      track.append(marker);
    }
    track.setAttribute("aria-hidden", "true");
    display.append(track);
  }
  return display;
}

function targetPatchesFor(view) {
  const annotation = state.case.localization;
  if (view.localization && annotation?.gridShape[0] === view.values.length
      && annotation.gridShape[1] === view.values[0].length
      && annotation.region === view.localization.region) {
    return new Set(annotation.targetPatches);
  }
  return undefined;
}

function spatialOptions(view, readoutId, reference = false) {
  return {
    reference, options: state.options, readoutId, updatePatch,
    currentPatch: key => state.patches.get(key), targetPatches: targetPatchesFor(view),
  };
}

function updatePatch(key, patch) {
  if (patch) state.patches.set(key, patch);
  else state.patches.delete(key);
  for (const map of state.maps.values()) {
    if (map.key === key) map.select(patch);
  }
}

function releaseMap(methodId) {
  state.maps.get(methodId)?.destroy();
  state.maps.delete(methodId);
}

function renderFigure(view) {
  const figure = node("figure");
  const photo = node("img", "figure-image");
  photo.src = view.image.src;
  photo.width = view.image.width;
  photo.height = view.image.height;
  photo.alt = view.image.alt || view.title;
  const link = node("a");
  link.setAttribute("aria-label", `Open full size: ${view.title}`);
  // Browsers disallow opening data URLs as top-level navigations.
  if (view.image.src.startsWith("data:")) {
    link.href = view.image.src;
    const extension = view.image.src.slice(view.image.src.indexOf("/") + 1, view.image.src.indexOf(";"));
    link.download = `probe-figure.${extension}`;
  } else {
    link.href = view.image.src;
    link.target = "_blank";
    link.rel = "noopener";
  }
  link.append(photo);
  figure.append(link);
  return figure;
}

function openReference(image) {
  const frame = node("div", "original-frame");
  const photo = node("img");
  photo.src = image.src;
  photo.alt = image.alt || "Original experiment image";
  frame.append(photo);
  for (const box of image.boxes || []) {
    const region = node("span", "original-box");
    region.style.left = `${100 * box.x / image.width}%`;
    region.style.top = `${100 * box.y / image.height}%`;
    region.style.width = `${100 * box.width / image.width}%`;
    region.style.height = `${100 * box.height / image.height}%`;
    if (box.label) region.append(node("span", "", box.label));
    frame.append(region);
  }
  ui.original.replaceChildren(frame);
  ui.referenceDialog.showModal();
}

function renderPanelBody(body, method, view, index) {
  if (view.type === "spatial") {
    const map = renderSpatial(view, view.image ?? state.case.image, spatialOptions(view, `patch-${index}`));
    state.maps.set(method.id, map);
    body.append(map.host);
  } else {
    if (method.views.length === 1) body.append(node("p", "view-title", view.title));
    body.append(view.type === "figure" ? renderFigure(view) : renderChart(view));
  }
  if (view.localization) {
    const settings = node("details", "map-method-details");
    settings.append(node("summary", "", "Map details"));
    if (method.description) settings.append(node("p", "", method.description));
    settings.append(node("p", "", `${view.title} · ${view.unit}`));
    if (view.localization.peakInRegion !== null && view.localization.peakInRegion !== undefined) {
      settings.append(node("p", "", view.localization.peakInRegion
        ? "The strongest patch is in the annotation."
        : "The strongest patch is outside the annotation."));
    }
    if (view.caption) settings.append(node("p", "", view.caption));
    body.append(settings);
    return;
  }
  const description = view.description ?? view.caption;
  if (description) body.append(node("p", "view-description", description));
}

function buildPanel(method, index, leaders) {
  const view = currentView(method);
  const panel = node("article", "panel");
  panel.dataset.method = method.id;
  panel.style.setProperty("--panel-accent", colors[index % colors.length]);
  if (leaders?.has(method.id)) panel.dataset.leader = "true";
  const header = node("header", "panel-header");
  const heading = node("div", "panel-heading");
  const accent = node("span", "panel-accent");
  accent.setAttribute("aria-hidden", "true");
  heading.append(accent, node("h3", "", method.label));
  if (view.localization) {
    if (Number.isFinite(view.layer)) heading.append(node("span", "metric-tag", `L${view.layer}`));
    const { zeroMap, massFraction, areaFraction } = view.localization;
    header.append(heading, shareDisplay(zeroMap ? null : massFraction, "signal at target", areaFraction));
  } else {
    if (view.unit) heading.append(node("span", "metric-tag", view.unit));
    header.append(heading);
    if (method.description) header.append(node("p", "panel-description", method.description));
  }
  panel.append(header);
  const body = node("div", "panel-body");
  const showMapFirst = state.family === "attention";
  if (showMapFirst) panel.append(body);

  if (method.views.length > 1) {
    const label = node("label", "panel-select", "View");
    const select = node("select");
    select.id = `view-for-method-${index}`;
    select.setAttribute("aria-label", `View for ${method.label}`);
    method.views.forEach((view, n) => select.add(new Option(view.title, n)));
    select.value = state.views.get(method.id) || 0;
    select.addEventListener("change", () => {
      state.views.set(method.id, Number(select.value));
      renderPanels();
      byId(select.id).focus({ preventScroll: true });
    });
    label.append(select);
    panel.append(label);
  } else if (view.type === "spatial" && !view.localization) {
    panel.append(node("p", "panel-select static-view", view.title));
  }
  renderPanelBody(body, method, view, index);
  if (!showMapFirst) panel.append(body);
  return panel;
}

function buildReference({ view, image }) {
  const panel = node("article", "panel reference-panel");
  panel.style.setProperty("--panel-accent", "#267963");
  const header = node("header", "panel-header");
  const heading = node("div", "panel-heading");
  heading.append(node("span", "panel-accent"), node("h3", "", "Reference image"));
  header.append(heading);
  if (view.localization) {
    header.append(shareDisplay(view.localization.areaFraction, "of patches touch box"));
  } else {
    header.append(node("p", "panel-description", "Model input"));
  }
  const body = node("div", "panel-body");
  const map = renderSpatial(view, image, spatialOptions(view, "reference-patch", true));
  state.maps.set(referenceKey, map);
  body.append(map.host);
  const original = state.case.image;
  if (original?.src === image.src && !original.content) {
    const button = node("button", "reference-open", "View original image");
    button.type = "button";
    button.addEventListener("click", () => openReference(original));
    body.append(button);
  }
  panel.append(header, body);
  return panel;
}

function renderCaseEvidence() {
  renderEvidence(ui.evidence, state.case.comparison, state.active, state.evidenceMetric, metric => {
    state.evidenceMetric = metric;
    renderCaseEvidence();
  });
}

function renderPanels() {
  for (const id of state.maps.keys()) releaseMap(id);
  const methods = activeMethods();
  const maps = mapInfo();
  const reference = state.family === "attention" ? maps.find(map => map.image) : null;
  const result = localizationComparison(maps);
  const panels = methods.map(method => buildPanel(method, state.case.methods.indexOf(method), result?.leaders));
  if (reference) panels.unshift(buildReference(reference));
  ui.summary.querySelector(".sample-media").hidden = !state.case.image || Boolean(reference);
  ui.panels.dataset.count = panels.length;
  ui.panels.dataset.family = state.family;
  ui.panels.style.setProperty("--columns", Math.min(panels.length, 4));
  ui.panels.replaceChildren(...panels);
  renderLocalization(result);
  renderCaseEvidence();
  updateToolbar(maps);
}

function mapInfo() {
  return activeMethods().flatMap(method => {
    const view = currentView(method);
    return view.type === "spatial" ? [{ method, view, image: view.image ?? state.case.image }] : [];
  });
}

function updateToolbar(maps) {
  ui.toolbar.hidden = !maps.length;
  ui.scaleNote.hidden = !maps.length;
  if (!maps.length) return;

  const canOverlay = maps.some(map => map.image);
  for (const button of ui.modes.children) button.hidden = !canOverlay && button.dataset.mode !== "heatmap";
  ui.boxesField.hidden = !maps.some(map => map.image?.boxes?.length);
  const layered = maps.filter(map => Number.isFinite(map.view.layer));
  const sharedLayers = layered.length ? [...new Set(layered[0].method.views
    .filter(view => view.type === "spatial" && Number.isFinite(view.layer)).map(view => view.layer))]
    .filter(layer => layered.every(({ method }) => method.views.some(view => view.type === "spatial" && view.layer === layer)))
    .sort((a, b) => a - b) : [];
  const canChangeLayer = sharedLayers.length > 1 || sharedLayers.length === 1
    && layered.some(({ view }) => view.layer !== sharedLayers[0]);
  ui.layerField.hidden = !canChangeLayer;
  if (canChangeLayer) {
    const currentLayer = layered[0].view.layer;
    const matched = sharedLayers.includes(currentLayer) && layered.every(({ view }) => view.layer === currentLayer);
    ui.layer.replaceChildren(new Option("Mixed", "", !matched, !matched));
    ui.layer.options[0].disabled = true;
    for (const layer of sharedLayers) ui.layer.add(new Option(`L${layer}`, layer, false, matched && currentLayer === layer));
  }
  updateMapDisplay(canOverlay);
}

function updateMapDisplay(canOverlay = mapInfo().some(map => map.image)) {
  const mode = canOverlay ? state.options.mode : "heatmap";
  for (const button of ui.modes.children) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  ui.opacityField.hidden = mode !== "overlay";
  ui.focusField.hidden = mode !== "focus";
  ui.scaleNote.replaceChildren();
  if (mode === "focus") {
    const gold = node("span", "map-key key-positive", "Positive");
    const blue = node("span", "map-key key-negative", "Negative");
    const peak = node("span", "map-key key-peak", "Peak patch");
    ui.scaleNote.append(gold, blue, peak);
    if (state.options.boxes && mapInfo().some(({ view }) => targetPatchesFor(view))) {
      ui.scaleNote.append(node("span", "map-key key-target", "Target patches"));
    }
    ui.scaleNote.append(node("span", "map-key-note", "Strongest patches covering the selected share. Ties are included."));
  } else if (mode !== "image") {
    ui.scaleNote.append(node("span", "", "Each map uses its own color scale."));
  }
  for (const map of state.maps.values()) map.refresh();
}

function selectCase(c, family = c.family) {
  if (state.family !== family) state.options.mode = family === "attention" ? "focus" : "overlay";
  state.family = family;
  state.case = c;
  state.views.clear();
  state.patches.clear();
  state.evidenceMetric = null;
  renderFamilySelection();
  if (!c) {
    for (const id of state.maps.keys()) releaseMap(id);
    ui.panels.replaceChildren();
    ui.summary.hidden = true;
    ui.comparison.hidden = true;
    notice("No experiments are loaded for this family. Import a result file to get started.");
    return;
  }
  const defaults = c.defaultMethods || c.methods.slice(0, 3).map(method => method.id);
  state.active = new Set(defaults.slice(0, methodLimit));
  renderSummary(c);
  renderMethodSelection();
  ui.comparison.hidden = false;
  renderPanels();
  renderSource(c);
}

function addCases(cases, imported) {
  const epoch = state.cases.length;
  const added = cases.map((c, index) => ({ ...c, key: `case-${epoch + index}`, imported }));
  state.cases.push(...added);
  return added;
}

renderFamilies();
ui.caseSelect.addEventListener("change", () => {
  const next = state.cases.find(c => c.key === ui.caseSelect.value);
  if (next) {
    notice("");
    selectCase(next);
  }
});
ui.modes.addEventListener("click", event => {
  const mode = event.target.closest("button")?.dataset.mode;
  if (!mode) return;
  state.options.mode = mode;
  updateMapDisplay();
});
ui.layer.addEventListener("change", () => {
  const layer = Number(ui.layer.value);
  for (const { method, view } of mapInfo()) {
    if (!Number.isFinite(view.layer)) continue;
    const next = method.views.findIndex(candidate => candidate.type === "spatial" && candidate.layer === layer);
    if (next >= 0) state.views.set(method.id, next);
  }
  renderPanels();
});
ui.opacity.addEventListener("input", () => {
  state.options.opacity = Number(ui.opacity.value) / 100;
  ui.opacityValue.textContent = `${ui.opacity.value}%`;
  updateMapDisplay();
});
ui.focus.addEventListener("change", () => {
  state.options.focusFraction = Number(ui.focus.value);
  updateMapDisplay();
});
ui.grid.addEventListener("change", () => {
  state.options.grid = ui.grid.checked;
  updateMapDisplay();
});
ui.boxes.addEventListener("change", () => {
  state.options.boxes = ui.boxes.checked;
  updateMapDisplay();
});
ui.importButton.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", async () => {
  const file = ui.file.files[0];
  if (!file) return;
  ui.file.value = "";
  try {
    const cases = validateBundle(JSON.parse(await file.text()), { imported: true });
    const added = addCases(cases, true);
    selectCase(added[0]);
    notice(`Imported ${added.length} ${added.length === 1 ? "experiment" : "experiments"} from ${file.name}.`);
  } catch (error) {
    notice(`Could not import ${file.name}. ${error.message}`, true);
  }
});

try {
  const response = await fetch("data/examples.json");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const added = addCases(validateBundle(await response.json()), false);
  if (!state.case) selectCase(added.find(c => c.family === state.family) || added[0]);
  else renderFamilySelection();
} catch (error) {
  renderFamilySelection();
  notice(`Archived examples could not load. Serve the repository over HTTP or import a result file. ${error.message}`, true);
}
