export const families = {
  attention: {
    label: "Attention", icon: "◉", color: "#365bcc",
    description: "Compare where attention and attribution concentrate on the same input.",
  },
  lens: {
    label: "Lens", icon: "◇", color: "#714eb3",
    description: "Trace which tokens become readable across layers and positions.",
  },
  causal: {
    label: "Causal", icon: "⇄", color: "#a2494d",
    description: "Compare how interventions change a selected answer score.",
  },
  sae: {
    label: "Sparse autoencoders", icon: "⠿", color: "#24766c",
    description: "Inspect sparse features, reconstruction fidelity, and the effects of feature edits.",
  },
};

function ensure(condition, context, expectation) {
  if (!condition) throw new Error(`${context} needs ${expectation}.`);
}

const nonempty = value => typeof value === "string" && value.trim().length > 0;
const scalar = value => value === null || Number.isFinite(value);
const label = value => typeof value === "string" || Number.isFinite(value);
const tableCell = value => scalar(value) || typeof value === "string" || typeof value === "boolean";
const seq = (value, valid) => Array.isArray(value) && value.length > 0 && value.every(valid);
const sameLength = (value, count, valid) => Array.isArray(value) && value.length === count && value.every(valid);
const localAsset = /^\.\.\/(?:demos\/assets\/images|docs\/assets\/tutorials)\/[\w./-]+\.(?:png|jpe?g|webp)$/i;
const dataImage = /^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=\r\n]+$/i;
const validPath = /^(?:demos|docs)\/[\w./-]+(?:#[\w-]+)?$/;

function within(box, image) {
  return box && [box.x, box.y, box.width, box.height].every(Number.isFinite)
    && box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0
    && box.x + box.width <= image.width + .01 && box.y + box.height <= image.height + .01;
}

function imageSpec(image, context, imported) {
  ensure(image && typeof image.src === "string" && (dataImage.test(image.src)
    || (!imported && localAsset.test(image.src) && !image.src.slice(3).split("/").includes(".."))),
  context, imported ? "an embedded PNG, JPEG, or WebP image" : "a supported image source");
  ensure(Number.isFinite(image.width) && image.width > 0 && Number.isFinite(image.height) && image.height > 0,
    context, "positive width and height");
  if (image.content !== undefined) ensure(within(image.content, image), context, "a content rectangle inside its frame");
  if (image.background !== undefined) ensure(/^#[a-f0-9]{6}$/i.test(image.background), context, "a hex background color");
  if (image.boxes !== undefined) {
    ensure(Array.isArray(image.boxes) && image.boxes.every(box => within(box, image)), context, "regions inside its frame");
  }
}

function validateView(view, context, imported) {
  ensure(view && nonempty(view.title), context, "a view title");
  if (view.image !== undefined) imageSpec(view.image, context, imported);
  if (view.bounds !== undefined) {
    ensure(sameLength(view.bounds, 2, Number.isFinite) && view.bounds[0] <= view.bounds[1], context, "ordered finite bounds");
  }
  switch (view.type) {
    case "spatial":
      ensure(seq(view.values, row => seq(row, scalar)) && view.values.every(row => row.length === view.values[0].length),
        context, "a rectangular grid of numbers or nulls");
      break;
    case "matrix":
      ensure(seq(view.rows, label) && seq(view.columns, label)
        && sameLength(view.values, view.rows.length, row => sameLength(row, view.columns.length, scalar)),
      context, "a matrix matching its row and column labels");
      break;
    case "series":
      ensure(seq(view.x, label) && seq(view.series, s => s && nonempty(s.label) && sameLength(s.values, view.x.length, scalar)),
        context, "traces with values for each x coordinate");
      break;
    case "bars":
      ensure(seq(view.entries, entry => entry && nonempty(entry.label) && scalar(entry.value)),
        context, "labeled entries with numeric or missing values");
      break;
    case "table":
      ensure(seq(view.columns, label) && seq(view.rows, row => sameLength(row, view.columns.length, tableCell)),
        context, "table rows matching the column labels");
      break;
    case "figure":
      ensure(view.image, context, "a figure image");
      break;
    default:
      throw new Error(`${context} has an unknown view type.`);
  }
}

export function validateBundle(bundle, { imported = false } = {}) {
  ensure(bundle?.schema === 1 && seq(bundle.cases, c => c && nonempty(c.id)), "Result file",
    "schema 1 and at least one experiment with an ID");
  const ids = new Set();
  for (const c of bundle.cases) {
    const context = `Experiment “${c.id}”`;
    ensure(!ids.has(c.id), context, "a unique ID");
    ids.add(c.id);
    ensure(Object.hasOwn(families, c.family) && nonempty(c.title), context, "a supported family and a title");
    if (c.image !== undefined) imageSpec(c.image, context, imported);
    ensure(seq(c.methods, method => method && nonempty(method.id) && nonempty(method.label) && seq(method.views, Boolean)),
      context, "labeled methods with at least one view");
    const methodIds = new Set();
    for (const method of c.methods) {
      const methodContext = `${c.title} / ${method.label}`;
      ensure(!methodIds.has(method.id), methodContext, "a unique method ID");
      methodIds.add(method.id);
      method.views.forEach((view, index) => validateView(view, `${methodContext} / view ${index + 1}`, imported));
    }
    if (c.defaultMethods !== undefined) {
      ensure(seq(c.defaultMethods, id => methodIds.has(id)), context, "default methods from this experiment");
    }
  }
  return bundle.cases;
}

export function sourceHref(source, repository, archived) {
  if (typeof source?.path !== "string" || !validPath.test(source.path) || source.path.split("/").includes("..")) return null;
  const ref = typeof source.commit === "string" && /^[a-f0-9]{7,40}$/i.test(source.commit)
    ? source.commit
    : archived ? "7e5aca79ecd154c88e34c4bfb2171352d80db675" : "main";
  return `${repository}/blob/${ref}/${source.path}`;
}
