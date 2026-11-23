import { spatialKey } from "./spatial.js";

const percent = new Intl.NumberFormat("en", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const points = new Intl.NumberFormat("en", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
  signDisplay: "exceptZero",
});

function annotationKey(image) {
  return (image?.boxes ?? [])
    .map(({ x, y, width, height }) => JSON.stringify([x, y, width, height]))
    .sort();
}

function scopeKey(view, image, layer, { region, areaFraction }) {
  return JSON.stringify([layer, spatialKey(view, image), annotationKey(image), region, areaFraction]);
}

function scoreRelation(score, baseline, displayed) {
  if (score === null) return "unscored";
  if (displayed === percent.format(baseline)) return "at";
  return score < baseline ? "below" : "above";
}

export function compareLocalization(maps, preferredLayer = null) {
  const scoped = new Map();
  for (const { method, view, image } of maps) {
    const measurement = view.localization;
    if (!measurement) continue;
    const layer = Number.isFinite(view.layer) ? view.layer : null;
    const { areaFraction, region, zeroMap, massFraction } = measurement;
    const key = scopeKey(view, image, layer, measurement);
    if (!scoped.has(key)) scoped.set(key, { areaFraction, region, layer, rows: [] });
    const score = zeroMap ? null : massFraction;
    const displayed = score === null ? "No signal" : percent.format(score);
    const relation = scoreRelation(score, areaFraction, displayed);
    scoped.get(key).rows.push({ method, view, score, displayed, relation, rank: null });
  }
  if (!scoped.size) return null;

  const leaders = new Set();
  const groups = [...scoped.values()];
  for (const group of groups) {
    group.rows.sort((a, b) => a.displayed === b.displayed ? 0 : (b.score ?? -1) - (a.score ?? -1));
    group.signalCount = group.rows.filter(row => row.score !== null).length;
    let previous = null;
    for (const [index, row] of group.rows.entries()) {
      if (row.score === null || group.signalCount < 2) continue;
      row.rank = previous?.displayed === row.displayed ? previous.rank : index + 1;
      previous = row;
    }
    group.winningRows = group.rows.filter(row => row.rank === 1);
    for (const { method } of group.winningRows) leaders.add(method.id);
  }
  const favored = group => Number(preferredLayer !== null && group.layer === preferredLayer);
  groups.sort((a, b) => favored(b) - favored(a)
    || b.rows.length - a.rows.length
    || (b.layer ?? -1) - (a.layer ?? -1));
  return { groups, leaders };
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function layerLabel(layer) {
  return layer === null ? "Recorded views" : `Layer ${layer} endpoint`;
}

function renderInsight(group) {
  const panel = node("div", "attention-rank-insight");
  const scope = node("div", "attention-rank-scope");
  const count = group.rows.length;
  scope.append(node("h4", "", layerLabel(group.layer)), node("span", "", `${count} ${count === 1 ? "map" : "maps"}`));
  panel.append(scope);

  const first = group.rows[0];
  const winners = group.winningRows;
  const insight = node("div", "attention-rank-highlight");
  if (!group.signalCount) {
    insight.append(node("strong", "attention-rank-highlight-text", "No signal"),
      node("span", "attention-rank-highlight-label", "The selected maps have zero signal."));
  } else {
    const name = winners.length > 1 ? `${winners.length} methods tied` : first.method.label;
    const label = winners.length ? "Highest target share" : "Measured target share";
    insight.append(
      node("span", "attention-rank-highlight-label", label),
      node("strong", "attention-rank-highlight-value", first.displayed),
      node("span", "attention-rank-highlight-name", name),
    );
  }
  panel.append(insight);

  const baseline = node("div", "attention-rank-baseline");
  baseline.append(node("span", "attention-rank-baseline-glyph"), node("strong", "", percent.format(group.areaFraction)),
    node("span", "", "uniform baseline"));
  panel.append(baseline);
  if (group.signalCount && group.rows.filter(row => row.score !== null).every(row => row.relation === "below")) {
    panel.append(node("p", "attention-rank-baseline-note", "All measured shares are below baseline."));
  }
  return panel;
}

function renderScoreRow(row, baseline, onInspect) {
  const item = node("li", "attention-rank-item");
  const button = node("button", "attention-rank-button");
  button.type = "button";
  button.dataset.relation = row.relation;
  button.dataset.method = row.method.id;
  if (row.rank === 1) button.dataset.leading = "true";
  const measured = row.score !== null;
  const rank = row.rank === null ? "" : `Rank ${row.rank}. `;
  const placement = measured ? ` ${row.relation} uniform baseline.` : "";
  button.setAttribute("aria-label", `${rank}${row.method.label}. ${row.displayed}${placement} Inspect map.`);
  button.addEventListener("click", () => onInspect(row.method.id));

  const position = node("span", "attention-rank-position", row.rank ?? "·");
  const title = node("span", "attention-rank-method", row.method.label);
  const value = node("span", "attention-rank-value");
  value.append(node("strong", "attention-rank-score", row.displayed));
  if (measured) {
    const delta = row.relation === "at" ? "At baseline" : `${points.format((row.score - baseline) * 100)} pp`;
    value.append(node("span", "attention-rank-delta", delta));
  }
  const track = node("span", "attention-rank-track");
  track.setAttribute("aria-hidden", "true");
  if (measured) {
    const fill = node("span", "attention-rank-fill");
    fill.style.inlineSize = `${row.score * 100}%`;
    track.append(fill, node("span", "attention-rank-marker"));
  }
  button.append(position, title, track, value);
  item.append(button);
  return item;
}

function renderGroup(group, onInspect) {
  const stage = node("div", "attention-rank-group");
  stage.setAttribute("role", "group");
  stage.setAttribute("aria-label", `${layerLabel(group.layer)} for the ${group.region} region`);
  stage.style.setProperty("--attention-rank-baseline", `${group.areaFraction * 100}%`);
  stage.append(renderInsight(group));

  const content = node("div", "attention-rank-content");
  const legend = node("div", "attention-rank-legend");
  legend.append(node("span", "", "Target share by method"));
  const below = node("span", "attention-rank-below-key", "Below baseline");
  below.hidden = !group.rows.some(row => row.relation === "below");
  legend.append(below);

  const list = node("ol", "attention-rank-list");
  list.classList.toggle("attention-rank-list-wide", group.rows.length > 4);
  list.style.setProperty("--attention-rank-rows", Math.ceil(group.rows.length / 2));
  for (const row of group.rows) list.append(renderScoreRow(row, group.areaFraction, onInspect));
  content.append(legend, list);
  stage.append(content);
  return stage;
}

export function renderAttentionSummary(container, result, onInspect) {
  container.replaceChildren();
  container.hidden = !result;
  if (!result) return;
  container.classList.add("attention-rank-board");
  const intro = node("header", "attention-rank-intro");
  const title = node("div", "attention-rank-title");
  const description = result.groups.length > 1
    ? "Absolute signal on patches touching the annotation. Each group shares a layer and patch grid."
    : "Share of absolute signal on patches touching the annotated box.";
  title.append(node("h3", "", "Target localization"), node("p", "", description));
  intro.append(title, node("p", "attention-rank-inspect-tip", "Select a method to inspect its map"));
  container.append(intro);
  for (const group of result.groups) container.append(renderGroup(group, onInspect));
}
