"""Held-out concept probes with train-only feature selection."""
import torch

from .types import EvaluationReport


def _matrix(value, name):
    value = torch.as_tensor(value).detach().double().cpu()
    if value.ndim != 2 or not len(value) or not value.shape[1] or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite nonempty [example,feature] matrix")
    return value


def _check_groups(train_ids, validation_ids, test_ids):
    supplied = [train_ids is not None, validation_ids is not None, test_ids is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("supply source group IDs for all three splits or none")
    if not any(supplied):
        return False
    groups = [set(map(str, values)) for values in (train_ids, validation_ids, test_ids)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("train, validation and test source groups must be disjoint")
    return True


def pca_features(X_train, X_validation, X_test, *, n_components=128):
    """Fit PCA ONLY on train; return transformed train, validation and test."""
    matrices = [_matrix(value, name) for value, name in zip(
        (X_train, X_validation, X_test), ("X_train", "X_validation", "X_test"))]
    if any(value.shape[1] != matrices[0].shape[1] for value in matrices):
        raise ValueError("all PCA splits must have the same input width")
    limit = min(matrices[0].shape)
    if type(n_components) is not int or not 1 <= n_components <= limit:
        raise ValueError("n_components must fit the training matrix dimensions")
    mean = matrices[0].mean(0)
    # Deterministic full SVD makes the transform independent of global RNG state.
    _, _, vectors = torch.linalg.svd(matrices[0] - mean, full_matrices=False)
    projection = vectors[:n_components].T
    return tuple((value - mean) @ projection for value in matrices)


def sparse_probe(X_train, y_train, X_validation, y_validation, X_test, y_test, *,
                 feature_counts=(1, 2, 5), regularization=(.01, .1, 1., 10.),
                 train_group_ids=None, validation_group_ids=None, test_group_ids=None,
                 seed=0, provenance=None):
    """Evaluate one-vs-rest sparse logistic probes without touching the SAE.

    For each class, select k features by absolute point-biserial correlation on
    TRAIN only; standardize using TRAIN statistics. Choose logistic regression C
    by validation AUROC, then report held-out test metrics. Validation/test never
    influence feature selection or scaling. This explicitly uses a development
    split, rather than calling development selection cross-validation.

    ``scikit-learn`` is imported only when this optional evaluation is requested.
    Compare SAE features, raw neurons, PCA or random features by passing each
    representation through this same function and identical split/group IDs.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    except ImportError as error:
        raise ImportError("sparse_probe requires scikit-learn; install vlmprobe[evaluation]") from error
    matrices = [_matrix(value, name) for value, name in zip(
        (X_train, X_validation, X_test), ("X_train", "X_validation", "X_test"))]
    if any(value.shape[1] != matrices[0].shape[1] for value in matrices):
        raise ValueError("probe feature widths must agree across splits")
    labels = [value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else
              value.tolist() if hasattr(value, "tolist") else list(value)
              for value in (y_train, y_validation, y_test)]
    if any(len(label) != len(matrix) for label, matrix in zip(labels, matrices)):
        raise ValueError("each label array must match its feature rows")
    classes = sorted(set(labels[0]), key=str)
    if len(classes) < 2 or any(set(label) - set(classes) for label in labels[1:]):
        raise ValueError("train needs at least two classes and must cover all evaluated labels")
    for split_name, split_labels in zip(("train", "validation", "test"), labels):
        if set(split_labels) != set(classes):
            raise ValueError(f"{split_name} must contain every class for one-vs-rest AUROC")
    if any(type(k) is not int or not 1 <= k <= matrices[0].shape[1] for k in feature_counts):
        raise ValueError("feature_counts must fit the feature width")
    if not feature_counts or not regularization or any(not np.isfinite(c) or c <= 0 for c in regularization):
        raise ValueError("feature_counts and positive regularization values are required")
    groups = _check_groups(train_group_ids, validation_group_ids, test_group_ids)
    if groups:
        for group_ids, matrix in zip((train_group_ids, validation_group_ids, test_group_ids), matrices):
            if len(group_ids) != len(matrix):
                raise ValueError("source group IDs must match each split size")
    mean, std = matrices[0].mean(0), matrices[0].std(0, correction=0)
    valid = std > 0
    std = std.clamp_min(1e-12)
    scaled = [(matrix - mean) / std for matrix in matrices]
    encoded = [np.asarray([classes.index(label) for label in split]) for split in labels]
    results, rows, selections = [], [], {}
    for k in feature_counts:
        probabilities = np.zeros((len(matrices[2]), len(classes)))
        aucs = []
        selected_for_k = {}
        for class_index, class_name in enumerate(classes):
            binary = [(values == class_index).astype(np.int64) for values in encoded]
            if any(len(np.unique(values)) < 2 for values in binary):
                raise ValueError("each split must contain positive and negative examples of every class")
            target = torch.from_numpy(binary[0]).double()
            correlation = ((scaled[0] * (target - target.mean())[:, None]).mean(0)).abs()
            correlation[~valid] = -float("inf")
            if int(valid.sum()) < k:
                raise ValueError("not enough nonconstant training features for requested feature budget")
            indices = correlation.argsort(descending=True, stable=True)[:k]
            design = [value[:, indices].numpy() for value in scaled]
            best = None
            for c in regularization:
                classifier = LogisticRegression(C=float(c), solver="liblinear", random_state=seed,
                                                max_iter=2000)
                classifier.fit(design[0], binary[0])
                auc = roc_auc_score(binary[1], classifier.predict_proba(design[1])[:, 1])
                candidate = (auc, -float(c))
                if best is None or candidate > best[0]:
                    best = (candidate, classifier, float(c))
            probability = best[1].predict_proba(design[2])[:, 1]
            probabilities[:, class_index] = probability
            auc = float(roc_auc_score(binary[2], probability))
            aucs.append(auc)
            selected_for_k[str(class_name)] = {"feature_ids": indices.tolist(), "C": best[2],
                "validation_auroc": float(best[0][0]), "test_auroc": auc}
        predicted = probabilities.argmax(1)
        results.append({"features_per_class": k, "macro_auroc": float(np.mean(aucs)),
            "macro_f1": float(f1_score(encoded[2], predicted, average="macro")),
            "accuracy": float(accuracy_score(encoded[2], predicted))})
        selections[str(k)] = selected_for_k
        for index, (truth, prediction) in enumerate(zip(encoded[2], predicted)):
            rows.append({"id": str(index), "features_per_class": k,
                         "label": str(classes[truth]), "prediction": str(classes[prediction]),
                         "correct": bool(truth == prediction)})
    return EvaluationReport({"n_train": len(matrices[0]), "n_validation": len(matrices[1]),
        "n_test": len(matrices[2]), "n_classes": len(classes), "results": results}, rows,
        provenance={**(provenance or {}), "evaluation_kind": "sparse_concept_probe",
            "feature_selection": "train-only absolute point-biserial correlation",
            "regularization_selection": "held-out validation AUROC, smaller C breaks ties",
            "standardization": "training mean and population standard deviation",
            "source_groups_checked": groups, "selected_features": selections,
            "feature_budget_definition": "k features separately selected per one-vs-rest class",
            "sae_and_backbone_frozen": True})
