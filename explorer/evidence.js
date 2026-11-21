import { renderChart } from "./charts.js";

const formatters = {
  percent: new Intl.NumberFormat("en", { style: "percent", maximumSignificantDigits: 3 }),
  count: new Intl.NumberFormat("en", { maximumFractionDigits: 0 }),
  decimal: new Intl.NumberFormat("en", { maximumSignificantDigits: 4 }),
  number: new Intl.NumberFormat("en", { maximumSignificantDigits: 5 }),
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function format(value, kind) {
  const formatter = Object.hasOwn(formatters, kind) ? formatters[kind] : formatters.number;
  return formatter.format(value);
}

function metricSelector(metrics, metric, onMetricChange) {
  const field = element("label", "evidence-metric-picker", "Metric");
  const select = element("select");
  for (const row of metrics) {
    const option = element("option", "", row.label);
    option.value = row.id;
    select.append(option);
  }
  select.value = metric.id;
  select.addEventListener("change", () => onMetricChange?.(select.value));
  field.append(select);
  return field;
}

function valueScale(entries) {
  let min = 0;
  let max = 0;
  for (const row of entries) {
    if (!Number.isFinite(row.value)) continue;
    min = Math.min(min, row.value);
    max = Math.max(max, row.value);
  }
  const denominator = Math.max(-min, max, Number.MIN_VALUE);
  const start = min / denominator;
  const extent = max / denominator - start;
  return {
    min,
    max,
    position(value) { return extent ? 100 * (value / denominator - start) / extent : 0; },
  };
}

function scoreRow(entry, metric, toPercent, badge) {
  const li = element("li", "evidence-score");
  const name = element("span", "evidence-score-label", entry.label);
  if (badge) {
    li.classList.add("is-leading");
    name.append(element("span", "evidence-rank", badge));
  }
  const rail = element("span", "evidence-rail");
  rail.setAttribute("aria-hidden", "true");
  const bar = element("span", "evidence-fill");
  const value = toPercent(entry.value);
  const zero = toPercent(0);
  if (zero > 0 && zero < 100) {
    rail.classList.add("is-signed");
    rail.style.setProperty("--zero", `${zero}%`);
  }
  bar.style.insetInlineStart = `${Math.min(zero, value)}%`;
  bar.style.inlineSize = `${Math.abs(zero - value)}%`;
  rail.append(bar);
  const displayedValue = element("strong", "evidence-score-value", format(entry.value, metric.format));
  displayedValue.title = String(entry.value);
  li.append(name, rail, displayedValue);
  return li;
}

function renderMetric(metric, activeMethods) {
  const wrapper = element("div", "evidence-measurement");
  const heading = element("div", "evidence-score-heading");
  const direction = metric.direction;
  const ranked = direction === "higher" || direction === "lower";
  heading.append(element("h4", "", metric.label));
  if (ranked) {
    const wording = direction === "higher" ? "Higher ranks first" : "Lower ranks first";
    heading.append(element("span", "evidence-criterion", wording));
  }
  wrapper.append(heading);
  if (metric.scope) wrapper.append(element("p", "evidence-scope", metric.scope));

  const entries = metric.entries.filter(entry => Number.isFinite(entry.value)
    && (!entry.method || activeMethods.has(entry.method)));
  if (!entries.length) {
    wrapper.append(element("p", "evidence-empty", "Select a method measured on this metric to see its score."));
    return wrapper;
  }

  const scores = element(ranked ? "ol" : "ul", "evidence-scores");
  scores.setAttribute("role", "list");
  if (ranked) entries.sort((left, right) => direction === "higher"
    ? right.value - left.value : left.value - right.value);

  const scale = valueScale(metric.entries);
  const bestValue = entries[0].value;
  const bestDisplayed = format(bestValue, metric.format);
  const tiedAtPrecision = ranked && entries.some(row => row.value !== bestValue
    && format(row.value, metric.format) === bestDisplayed);
  const tiedExactly = ranked && entries.filter(row => row.value === bestValue).length > 1;
  const bestLabel = direction === "higher" ? "Highest" : "Lowest";
  for (const entry of entries) {
    let badge = "";
    if (ranked && entries.length > 1) {
      if (tiedAtPrecision && format(entry.value, metric.format) === bestDisplayed) badge = "Tied at shown precision";
      else if (entry.value === bestValue) badge = tiedExactly ? `Joint ${bestLabel.toLowerCase()}` : bestLabel;
    }
    scores.append(scoreRow(entry, metric, scale.position, badge));
  }

  wrapper.append(scores);
  if (scale.min !== scale.max) {
    const [from, to] = [scale.min, scale.max].map(value => format(value, metric.format));
    wrapper.append(element("p", "evidence-scale", `Shared bar scale ${from} to ${to}`));
  }
  if (metric.description) wrapper.append(element("p", "evidence-note", metric.description));
  return wrapper;
}

function renderFacts(facts) {
  const list = element("dl", "evidence-facts");
  for (const fact of facts) {
    const card = element("div", "evidence-fact");
    card.append(element("dt", "", fact.label), element("dd", "evidence-fact-value", fact.value));
    if (fact.description) card.append(element("dd", "evidence-note", fact.description));
    list.append(card);
  }
  return list;
}

/** Present only matched measurements from the supplied experiment. */
export function renderEvidence(container, comparison, activeMethodIds, metricId, onMetricChange) {
  const restoreFocus = container.contains(document.activeElement)
    && document.activeElement?.matches(".evidence-metric-picker select");
  container.replaceChildren();
  container.hidden = !comparison;
  if (!comparison) return;

  const methods = new Set(activeMethodIds);
  const metrics = comparison.metrics || [];
  const metric = metrics.find(row => row.id === metricId) || metrics[0];
  const board = element("div", "evidence-board");
  const header = element("div", "evidence-header");
  const introductory = element("div", "evidence-intro");
  introductory.append(element("h3", "", comparison.title));
  if (comparison.description) introductory.append(element("p", "", comparison.description));
  header.append(introductory);
  if (metrics.length > 1) header.append(metricSelector(metrics, metric, onMetricChange));
  board.append(header);
  const body = element("div", "evidence-body");
  const summary = element("div", "evidence-summary");
  const facts = (comparison.facts || []).filter(fact => !fact.method || methods.has(fact.method));
  if (facts.length) summary.append(renderFacts(facts));
  if (metric) summary.append(renderMetric(metric, methods));
  if (summary.childElementCount) body.append(summary);
  if (comparison.chart) {
    const chart = element("div", "evidence-chart");
    chart.append(element("h4", "", comparison.chart.title), renderChart(comparison.chart));
    if (comparison.chart.description) chart.append(element("p", "evidence-note", comparison.chart.description));
    body.append(chart);
    body.classList.toggle("is-paired", summary.childElementCount > 0);
  }
  if (body.childElementCount) board.append(body);
  container.append(board);
  if (restoreFocus) container.querySelector(".evidence-metric-picker select")?.focus({ preventScroll: true });
}
