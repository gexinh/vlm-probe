"""Build the browser examples from repository measurements, without model weights."""

import ast
import json
from math import exp
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "explorer/data/examples.json"
SNAPSHOTS = "demos/snapshots"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def image(path, size, boxes=(), **geometry):
    if not (ROOT / path).is_file():
        raise ValueError(f"Missing image: {path}")
    width, height = size
    return {
        "src": "../" + path,
        "width": width,
        "height": height,
        "boxes": [
            dict(zip(("label", "x", "y", "width", "height"), (label, *coords)))
            for label, coords in boxes
        ],
        **geometry,
    }


def gqa_image(sample, *, padding=None):
    path = "demos/assets/images/" + Path(sample["image_path"]).name
    width, height = sample["image_size"]
    label = sample["object_name"]
    x, y, box_width, box_height = sample["bbox_xywh"]
    if padding is None:
        return image(path, (width, height), [(label, (x, y, box_width, box_height))])
    # LLaVA-1.5 pads to square using the CLIP mean, then uniformly resizes.
    # Draw the original JPEG within the same padded coordinates as the native grid.
    left, top = padding
    side = max(width, height)
    return image(
        path, (side, side), [(label, (x + left, y + top, box_width, box_height))],
        content={"x": left, "y": top, "width": width, "height": height},
        background="#7a7468",
    )


def source(path, description):
    return {"path": path, "description": description}


def method(key, label, description, *views):
    return {"id": key, "label": label, "description": description, "views": list(views)}


def matrix(title, layers, columns, values, *, unit, signed=False, **details):
    return {
        "type": "matrix", "title": title, "unit": unit, "signed": signed,
        "rows": [f"L{layer}" for layer in layers], "columns": columns,
        "values": values, **details,
    }


def series(title, x, traces, *, unit, signed=False, **details):
    return {
        "type": "series", "title": title, "unit": unit, "signed": signed,
        "x": x, "series": [{"label": label, "values": values} for label, values in traces],
        **details,
    }


def table(title, columns, rows, **details):
    return {"type": "table", "title": title, "columns": columns, "rows": rows, **details}


def comparison_metric(key, label, formatting, direction, entries, *, scope):
    return {
        "id": key, "label": label, "format": formatting, "direction": direction,
        "entries": entries, "scope": scope,
    }


def metric_entry(label, value, method_id=None):
    entry = {"label": label, "value": value}
    if method_id is not None:
        entry["method"] = method_id
    return entry


ATTENTION_METHODS = {
    "Raw attention": (
        "raw-attention", "Raw attention", "attention probability",
        "Native decoder attention averaged across heads and answer-prediction positions.",
    ),
    "Grad-CAM": (
        "grad-cam", "Grad-CAM", "CAM score",
        "Reweights the attention heads by their gradients for the fixed answer score.",
    ),
    "Generic relevance": (
        "relevance", "Attention relevance", "relevance score",
        "Propagates positive gradient-weighted attention for the fixed answer score.",
    ),
    "Rollout": (
        "rollout", "Attention rollout", "transition weight",
        "Composes residual self-attention transitions through the decoder stack.",
    ),
    "ATTATTR": (
        "attattr", "ATTATTR", "signed attribution",
        "Integrates gradients along the selected layer's zero-to-observed attention path.",
    ),
    "TAM": (
        "tam", "TAM", "attribution score",
        "Combines backward attention transitions with integrated feedback along an image path.",
    ),
    "Beyond Intuition-H": (
        "beyond-head", "Beyond Intuition H", "attribution score",
        "Combines head-weighted perception with image-path reasoning feedback.",
    ),
    "Beyond Intuition-C": (
        "beyond-token", "Beyond Intuition C", "attribution score",
        "Combines token-weighted perception with image-path reasoning feedback.",
    ),
}


def attention_example(question_id):
    path = f"{SNAPSHOTS}/vlm_attention_comparison/{question_id}.json"
    result = read(path)
    sample = result["sample"]
    processed = gqa_image(sample, padding=result["input"]["image_pad_offset_xy"])
    region = sample["object_name"]
    token_positions = result["input"]["token_positions"]
    target_positions = set(token_positions["target"])
    # Translate the snapshot's expanded-token mask to flat, row-major patch coordinates.
    target_patches = {index for index, position in enumerate(token_positions["image"]) if position in target_positions}
    methods = []
    for name, (key, label, unit, description) in ATTENTION_METHODS.items():
        record = result["methods"][name]
        layers = record["metadata"]["layers"]
        maps = record["maps"]
        diagnostics = record["spatial_diagnostics"]
        views = []
        map_layers = layers if len(maps) == len(layers) else layers[-1:]
        for layer, values, diagnostic in reversed(list(zip(map_layers, maps, diagnostics))):
            title = f"Decoder layer {layer}" if len(maps) > 1 else f"Through decoder layer {layer}"
            flat = [abs(value) for row in values for value in row]
            peak_value = max(flat)
            peak_patches = {idx for idx, value in enumerate(flat) if value == peak_value} if peak_value else set()
            peak_in_region = None
            if peak_patches:
                if peak_patches <= target_patches:
                    peak_in_region = True
                elif peak_patches.isdisjoint(target_patches):
                    peak_in_region = False
            views.append({
                "type": "spatial", "title": title, "layer": layer, "values": values,
                "unit": unit, "signed": record["signed"], "image": processed,
                "localization": {
                    "region": region, "massFraction": diagnostic["shirt_absolute_mass_fraction"],
                    "areaFraction": diagnostic["shirt_patch_area_fraction"],
                    "negativeMassFraction": diagnostic["negative_mass_fraction"],
                    "zeroMap": diagnostic["zero_map"], "peakInRegion": peak_in_region,
                },
            })
            if name in {"TAM", "Beyond Intuition-H", "Beyond Intuition-C"}:
                views[-1]["caption"] = "Causal decoder masks can favor early positions during propagation."
        if len(maps) > 1:
            views.append(series(
                "Map mass inside the shirt", layers,
                [("Share of absolute map mass", [d["shirt_absolute_mass_fraction"] for d in diagnostics]),
                 ("Shirt share of grid", [d["shirt_patch_area_fraction"] for d in diagnostics])],
                unit="fraction", xLabel="Decoder layer",
            ))
        methods.append(method(key, label, description, *views))

    color = sample["answer"]
    return {
        "id": f"gqa-{color}-attention", "family": "attention",
        "title": f"GQA · {color.capitalize()} shirt",
        "model": result["model_id"], "dataset": "GQA validation",
        "prompt": sample["question"], "target": result["target_text"],
        "metric": "Complete-answer log probability for target-conditioned methods",
        "description": "Compare the same decoder attention queries and 24 × 24 image patch grid.",
        "image": gqa_image(sample),
        "localization": {
            "region": region, "targetPatches": sorted(target_patches),
            "gridShape": result["input"]["geometry"]["grid_shape"],
            "areaFraction": len(target_patches) / len(token_positions["image"]),
        },
        "source": source(path, "Archived LLaVA report measurement on a bundled GQA image."),
        "defaultMethods": ["raw-attention", "grad-cam", "relevance"],
        "methods": methods,
    }


def lens_example():
    path = f"{SNAPSHOTS}/vlm_lens/07302654.json"
    result = read(path)
    sample = result["sample"]
    lens_readouts = [result[key] for key in ("logit", "tuned")]
    rank_bounds = [1, max(rank for readout in lens_readouts for row in readout["target_ranks"] for rank in row)]
    answer_bounds = [min(min(readout["full_answer_log_probability"]) for readout in lens_readouts), 0]
    shared_layers = sorted(set(lens_readouts[0]["layers"]) & set(lens_readouts[1]["layers"]))
    shared_layer_labels = ", ".join(f"L{layer}" for layer in shared_layers)
    shared_ranks = []
    shared_probabilities = []
    methods = []
    for key, label, description in [
        ("logit", "Logit Lens", "Reads intermediate residuals through LLaVA's native vocabulary head."),
        ("tuned", "Tuned Lens", "Uses affine layer translators fitted on separate COCO images."),
    ]:
        readout = result[key]
        columns = [f"{piece} · {position}" for piece, position in zip(readout["source_pieces"], readout["positions"])]
        answer_pieces = [readout["target_pieces"][index] for index in readout["answer_columns"]]
        shared_indices = [readout["layers"].index(layer) for layer in shared_layers]
        shared_probabilities.append(metric_entry(
            label, exp(readout["full_answer_log_probability"][shared_indices[-1]]), key,
        ))
        shared_ranks.append(metric_entry(
            label, median(readout["target_ranks"][row][col] for row in shared_indices for col in readout["answer_columns"]), key,
        ))
        methods.append(method(key, label, description,
            matrix(
                "Next-token rank across layers", readout["layers"], columns, readout["target_ranks"],
                unit="rank", lowerIsBetter=True, bounds=rank_bounds,
                rowLabel="Decoder layer", columnLabel="Source token · position",
                description=("Each position ranks its own teacher-forced next piece. "
                             f"The final columns score the answer pieces {' + '.join(answer_pieces)}."),
            ),
            series(
                "Complete-answer log probability", readout["layers"],
                [(result["target_text"], readout["full_answer_log_probability"])],
                unit="log probability", bounds=answer_bounds, xLabel="Decoder layer",
                description="Sum of both answer-piece log probabilities under their own teacher-forced prefixes.",
            ),
        ))

    embed = result["embed"]
    methods.append(method(
        "embed", "EmbedLens", "Finds the closest vocabulary embeddings to projected image patches.",
        {
            "type": "spatial", "title": "Best lexical cosine per patch",
            "values": embed["best_cosine"], "unit": "cosine similarity", "signed": False,
            "image": gqa_image(sample, padding=result["input"]["image_pad_offset_xy"]),
        },
        table("Nearest fragments inside the shirt", ["Patch", "Position", "Nearest fragment", "Cosine"],
              [[row["patch_index"], row["expanded_position"], row["pieces"][0], row["cosines"][0]]
               for row in embed["top_shirt_fragments"]]),
    ))
    patchscope = result["patchscope"]
    target = patchscope["words"][result["target_text"]]
    layers = patchscope["source_layers"]
    methods.append(method(
        "patchscope", "Patchscope", "Transfers a source residual into a separate color-completion prompt.",
        series(
            "Target-prompt answer log probability", layers,
            [
                ("Patched", target["patched_log_probabilities"]),
                ("Zero-vector control", target["zero_vector_log_probabilities"]),
                ("Target-prompt baseline", [target["baseline_log_probability"]] * len(layers)),
            ],
            unit="log probability", xLabel="Source decoder layer",
            description="Complete answer scored in the target prompt with the source answer-position residual inserted.",
        ),
        table(
            "Target prompt", ["Role", "Value"],
            [["Prompt", patchscope["target_prompt"]], ["Source position", patchscope["source_position"]],
             ["Target decoder layer", patchscope["target_layer"]]],
        ),
    ))
    return {
        "id": "gqa-yellow-lens", "family": "lens", "title": "GQA · decoding Yellow",
        "model": result["model_id"], "dataset": "GQA validation", "prompt": sample["question"],
        "target": result["target_text"], "metric": "Rank, answer log probability or lexical cosine",
        "description": "Read vocabulary evidence from text states and projected visual embeddings.",
        "image": gqa_image(sample),
        "comparison": {
            "title": f'Which lens recovers “{result["target_text"]}”?',
            "description": "Logit and Tuned Lens read the same answer pieces at the same decoder positions.",
            "metrics": [
                comparison_metric(
                    "answer-probability", "Complete-answer probability", "percent", "higher",
                    shared_probabilities, scope=f"L{shared_layers[-1]} · deepest shared layer",
                ),
                comparison_metric(
                    "answer-piece-rank", "Median answer-piece rank", "number", "lower", shared_ranks,
                    scope=f"Both answer pieces across {shared_layer_labels}",
                ),
            ],
        },
        "source": source(path, "Archived LLaVA lens readouts on an unseen GQA example."),
        "defaultMethods": ["logit", "tuned"],
        "methods": methods,
    }


def causal_example():
    path = f"{SNAPSHOTS}/activation_patching/llava.json"
    result = read(path)
    case = result["case"]
    actual_layers = result["actual_layers"]
    estimated_layers = result["layers"]
    measured = result["actual"]
    estimated = result["approximate"]
    estimated_indices = [estimated_layers.index(layer) for layer in actual_layers]
    aligned_estimates = {
        scope: [estimated[scope][index] for index in estimated_indices]
        for scope in ("visual", "prediction")
    }
    errors = {
        scope: [abs(exact - estimate) for exact, estimate in zip(measured[scope], aligned_estimates[scope])]
        for scope in ("visual", "prediction")
    }
    largest_image_error = max(range(len(actual_layers)), key=errors["visual"].__getitem__)
    measured_at_max = measured["visual"][largest_image_error]
    estimated_at_max = aligned_estimates["visual"][largest_image_error]
    max_score_change = max(
        abs(effects[scope][layers.index(layer)])
        for layers, effects in ((actual_layers, measured), (estimated_layers, estimated))
        for scope in ("visual", "prediction")
        for layer in actual_layers
    )
    max_token_change = max(abs(value) for effects in (measured, estimated) for row in effects["text"] for value in row)
    positions = [
        f"{piece} · {position}"
        for piece, position in zip(result["text_pieces"], result["text_positions"])
    ]
    controls = table(
        "Reference scores",
        ["Run", "Answer log probability"],
        [["Original image", result["baseline_log_probability"]],
         ["Gray donor", result["blank_donor_log_probability"]]],
    )
    methods = []
    for key, label, description, layers, effects in [
        ("patch", "Activation patching", "Replaces receiver residuals with gray-donor residuals and reruns the model.", actual_layers, measured),
        ("attribute", "Attribution patching", "Estimates the residual replacement effect from activation differences and local gradients.", estimated_layers, estimated),
    ]:
        # Keep both default charts aligned at the seven layers where exact replacements were run.
        indices = [layers.index(layer) for layer in actual_layers]
        methods.append(method(key, label, description,
            series(
                "Answer-score change", actual_layers,
                [
                    ("Image patches", [effects["visual"][index] for index in indices]),
                    ("Answer positions", [effects["prediction"][index] for index in indices]),
                ],
                unit="Δ log probability", signed=True, xLabel="Decoder layer",
                bounds=[-max_score_change, max_score_change],
                description="Change from the real-image baseline, evaluated at the layers with exact patch measurements.",
            ),
            matrix(
                "Change by text token", layers, positions, effects["text"],
                unit="Δ log probability", signed=True, bounds=[-max_token_change, max_token_change],
                rowLabel="Decoder layer", columnLabel="Token · position",
            ),
            controls,
        ))
    return {
        "id": "gqa-yellow-causal", "family": "causal", "title": "GQA · image replacement",
        "model": result["model"]["id"], "dataset": "GQA validation", "prompt": case["question"],
        "target": result["target"]["text"], "metric": "Change in complete-answer log probability",
        "description": "Compare exact and gradient-estimated interventions with a matched gray-image donor.",
        "image": gqa_image(case),
        "comparison": {
            "title": "How close is the gradient estimate?",
            "description": "Activation patching reruns the model. Its measured effects serve as the reference for the gradient estimate.",
            "metrics": [
                comparison_metric(
                    "max-gap", "Image-patch effect at the largest gap", "decimal", "descriptive",
                    [
                        metric_entry("Measured", measured_at_max, "patch"),
                        metric_entry("Gradient estimate", estimated_at_max, "attribute"),
                    ],
                    scope=f"L{actual_layers[largest_image_error]} · change in answer log probability",
                ),
                comparison_metric(
                    "patch-error", "Mean absolute estimation error", "decimal", "descriptive",
                    [
                        metric_entry("Image patches", mean(errors["visual"])),
                        metric_entry("Answer positions", mean(errors["prediction"])),
                    ],
                    scope=f"Answer log probability · {len(actual_layers)} shared layers",
                ),
            ],
            "chart": series(
                "Image-patch replacement by layer", actual_layers,
                [
                    ("Measured", measured["visual"]),
                    ("Gradient estimate", aligned_estimates["visual"]),
                ],
                unit="Δ log probability", signed=True, xLabel="Decoder layer",
            ),
        },
        "source": source(path, "Archived matched activation and attribution patching on the bundled GQA input."),
        "methods": methods,
    }


def notebook_output(notebook, cell_id):
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
    return "".join("".join(output["text"]) for output in cell["outputs"] if "text" in output)


def printed_mapping(output, label):
    return ast.literal_eval(next(line[len(label):] for line in output.splitlines() if line.startswith(label)))


def sae_example():
    path = "demos/11_sae_pretrained.ipynb"
    notebook = read(path)
    manifest = read("demos/assets/sae_report_samples.json")
    report = notebook_output(notebook, "4f960124")
    localization = printed_mapping(report, "Localization: ")
    reconstruction = printed_mapping(report, "Reconstruction/sparsity: ")
    interventions = json.loads(notebook_output(notebook, "700a2e7e"))
    images_by_id = {record["id"]: record for record in manifest["showcase"]}
    samples = [images_by_id[row["id"]] for row in interventions]
    categories = [manifest["categories"][str(record["category_id"])] for record in samples]
    feature_labels = [f"{category} · f{row['feature_id']}" for category, row in zip(categories, interventions)]
    car = next(record for category, record in zip(categories, samples) if category == "car")
    thumbnail = image(
        "demos/assets/images/sae_report/" + car["file_name"], car["image_size_wh"],
        [("car", box["bbox_xywh"]) for box in car["annotations"] if box["category_id"] == car["category_id"]],
    )
    figure = image("docs/assets/tutorials/sae-pretrained-dino-features.png", (1491, 1280))
    peak_hits = round(localization["point_at_1"] * localization["n_examples"])
    deletion_distance = mean(1 - row["selected_deletion_output_cosine"] for row in interventions)
    reconstruction_distance = mean(1 - row["reconstructed_output_cosine"] for row in interventions)
    return {
        "id": "dino-sae", "family": "sae", "title": "DINO · pretrained visual features",
        "model": "facebook/dino-vitb16", "dataset": "COCO 2017 validation",
        "prompt": "Inspect the selected person-, car- and dog-associated patch features.",
        "target": "Layer 11 · Prisma ReLU dictionary", "metric": "Patch localization and cosine fidelity",
        "description": "Features were selected on 32 independent calibration images and evaluated on three held-out photos.",
        "image": thumbnail,
        "comparison": {
            "title": "Do the selected features track the annotated objects?",
            "description": "The frozen person, car and dog features were selected on separate images.",
            "facts": [
                {
                    "label": "Strongest patch inside the object",
                    "value": f"{peak_hits} of {localization['n_examples']}",
                    "description": "One selected feature per held-out image.",
                },
                {
                    "label": "Activation inside the annotated region",
                    "value": f"{localization['mass_inside_active']:.1%}",
                    "description": "Average share across the active feature maps.",
                },
            ],
            "metrics": [
                comparison_metric(
                    "cosine-distance", "Change in the native representation", "decimal", "descriptive",
                    [
                        metric_entry("Delete selected feature", deletion_distance * 10_000),
                        metric_entry("Reconstruction control", reconstruction_distance * 10_000),
                    ],
                    scope="Cosine distance × 10⁴ · mean over three held-out images",
                ),
            ],
        },
        "source": source(path, "Executed DINO notebook, pinned Prisma weights, and published activation figure."),
        "defaultMethods": ["relu-features", "relu-edit"],
        "methods": [
            method(
                "relu-features", "Feature localization", "Inspect three frozen ReLU features in the model's actual input crop.",
                {
                    "type": "figure", "title": "Person, car and dog feature maps", "image": figure,
                    "description": "Published notebook output. Each row shows the original image, the 224 × 224 model crop and the 14 × 14 activation map.",
                },
                table(
                    "Held-out feature evaluation", ["Metric", "Measured", "Examples"],
                    [
                        ["Any active patch", localization["coverage"], localization["n_examples"]],
                        ["Peak in the annotated region", localization["point_at_1"], localization["n_examples"]],
                        ["Activation mass inside the region", localization["mass_inside_active"], localization["n_active"]],
                    ],
                ),
            ),
            method(
                "relu-edit", "Feature deletion", "Zeroes the selected feature while preserving the reconstruction error.",
                series(
                    "Change in native representation", feature_labels,
                    [
                        ("Feature deletion", [1 - row["selected_deletion_output_cosine"] for row in interventions]),
                        ("Reconstruction control", [1 - row["reconstructed_output_cosine"] for row in interventions]),
                    ],
                    unit="1 − cosine fidelity", xLabel="Held-out image · feature",
                ),
                table(
                    "Pooled patch representation change", ["Feature", "Patch L2 change", "CLS max absolute change"],
                    [[label, row["deletion_pooled_patch_l2_change"], row["deletion_cls_max_abs_change"]]
                     for label, row in zip(feature_labels, interventions)],
                ),
            ),
            method(
                "relu-reconstruction", "Reconstruction", "Checks how the frozen dictionary reconstructs held-out DINO patch activations.",
                table("Reconstruction and sparsity", ["Metric", "Measured"], [
                    ["Patch positions", reconstruction["n_positions"]], ["MSE", reconstruction["mse"]],
                    ["Fraction of variance unexplained", reconstruction["fvu"]],
                    ["Mean active features (L0)", reconstruction["l0"]],
                ]),
            ),
        ],
    }


SAE_LABELS = {
    "relu": "ReLU", "topk": "TopK", "gated": "Gated", "jumprelu": "JumpReLU",
    "abstopk": "AbsTopK", "batchtopk": "BatchTopK", "matryoshka_batchtopk": "Matryoshka BatchTopK",
    "matching_pursuit": "Matching Pursuit", "archetypal_topk": "Archetypal TopK",
    "archetypal_jumprelu": "Archetypal JumpReLU", "signed_matching_pursuit": "Signed MP",
    "orthogonal_matching_pursuit": "Nonnegative OMP", "signed_orthogonal_matching_pursuit": "Signed OMP",
    "qsae": "QSAE",
}


SAE_RECIPE_METRICS = [
    ("validation_fvu", "Variance unexplained (FVU)", "percent", "Validation split · lower is better"),
    ("mean_test_l0", "Active features per token (L0)", "decimal", "Test split · fewer active features means sparser codes"),
    ("validation_mse", "Reconstruction MSE", "decimal", "Validation split · lower is better"),
]


def sae_recipes_example():
    path = "docs/assets/tutorials/sae-examples-provenance.json"
    records = read(path)["methods"]
    return {
        "id": "sae-recipes", "family": "sae", "title": "SAE · dictionary recipes",
        "model": "Small randomly initialized ViT", "dataset": "Repository CPU tutorials",
        "prompt": "Compare sparsity and reconstruction in the repository's tutorial runs.",
        "metric": "Validation FVU, validation MSE and test L0",
        "description": "All 14 recipes use the same small illustrative setup. The runs measure mechanics and reconstruction.",
        "comparison": {
            "title": "Reconstruction quality versus sparsity",
            "description": "Compare selected recipes on the same small ViT tutorial setup. Lower FVU means better reconstruction and lower L0 means sparser codes.",
            "metrics": [
                comparison_metric(
                    key, label, formatting, "lower",
                    [metric_entry(SAE_LABELS[row["method"]], row[key], row["method"]) for row in records],
                    scope=scope,
                )
                for key, label, formatting, scope in SAE_RECIPE_METRICS
            ],
        },
        "source": source(path, "Execution measurements from the repository's validated CPU examples."),
        "defaultMethods": ["signed_orthogonal_matching_pursuit", "batchtopk", "matching_pursuit"],
        "methods": [
            method(
                row["method"], SAE_LABELS[row["method"]],
                f"Measured after {row['training_steps']} optimizer steps on {row['training_positions']} training positions.",
                table(
                    "Sparsity and reconstruction", ["Measure", "Value", "Split"],
                    [
                        ["Mean active features (L0)", row["mean_test_l0"], "Test"],
                        ["Fraction of variance unexplained", row["validation_fvu"], "Validation"],
                        ["Reconstruction MSE", row["validation_mse"], "Validation"],
                        ["Validation positions", row["validation_positions"], "Validation"],
                    ],
                ),
            )
            for row in records
        ],
    }


def main():
    examples = {
        "schema": 1,
        "cases": [
            attention_example("07302654"), attention_example("19225230"),
            sae_example(), sae_recipes_example(), lens_example(), causal_example(),
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(examples, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
