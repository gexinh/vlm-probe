"""Streaming float64 moments; batch boundaries do not define the metric."""
from collections import OrderedDict
from itertools import groupby
import hashlib
import json

import torch


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = None

    def add(self, values):
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if not len(values):
            return
        n = len(values)
        mean = values.mean(0)
        m2 = ((values - mean) ** 2).sum(0)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mean, m2
            return
        difference = mean - self.mean
        total = self.n + n
        self.m2 += m2 + difference.square() * (self.n * n / total)
        self.mean += difference * (n / total)
        self.n = total


class ReconstructionStatistics:
    def __init__(self, reference_mean=None):
        self.reference_mean = (None if reference_mean is None else
                               torch.as_tensor(reference_mean).detach().double().cpu())
        self.activations, self.features = Moments(), Moments()
        self.sse = self.energy = self.l0 = 0.
        self.reference_tss = 0.
        self.n_values = 0
        self.firing_count = self.minimum = self.maximum = None
        self.examples = OrderedDict()
        self.sample_digest = hashlib.sha256()
        self.activation_digest = hashlib.sha256()

    def add(self, values, features, reconstruction, ids, group_ids):
        values = values.detach().double().cpu()
        features = features.detach().double().cpu()
        reconstruction = reconstruction.detach().double().cpu()
        if len(values) != len(ids) or len(ids) != len(group_ids):
            raise ValueError("IDs must identify every selected activation position")
        # Canonical float64 row bytes authenticate actual cached observations.
        # Hash updates concatenate across chunks, so partitions do not matter.
        contiguous = values.contiguous()
        try:
            raw_bytes = contiguous.numpy().tobytes(order="C")
        except RuntimeError:  # PyTorch can run without optional NumPy installed.
            raw_bytes = bytes(contiguous.clone().untyped_storage())
        self.activation_digest.update(raw_bytes)
        error = (values - reconstruction).square().sum(-1)
        energy = values.square().sum(-1)
        l0 = features.ne(0).sum(-1)
        self.activations.add(values)
        self.features.add(features)
        self.sse += error.sum().item()
        self.energy += energy.sum().item()
        self.l0 += l0.sum().item()
        self.n_values += values.numel()
        if self.reference_mean is not None:
            if self.reference_mean.shape != (values.shape[-1],) or not torch.isfinite(self.reference_mean).all():
                raise ValueError("reference_mean must be a finite vector matching native activation width")
            self.reference_tss += (values - self.reference_mean).square().sum().item()
        counts = features.ne(0).sum(0)
        low, high = features.amin(0), features.amax(0)
        if self.firing_count is None:
            self.firing_count, self.minimum, self.maximum = counts, low, high
        else:
            if counts.shape != self.firing_count.shape:
                raise ValueError("dictionary width changed between batches")
            self.firing_count += counts
            self.minimum = torch.minimum(self.minimum, low)
            self.maximum = torch.maximum(self.maximum, high)
        # Consecutive selected tokens share a document/image ID. Aggregate
        # them together rather than performing Python work for every token.
        for (sample_id, group_id), run in groupby(enumerate(zip(ids, group_ids)), key=lambda item: item[1]):
            indices = [item[0] for item in run]
            count = len(indices)
            self.sample_digest.update((json.dumps((sample_id, group_id), ensure_ascii=False).encode() + b"\n") * count)
            if sample_id not in self.examples:
                self.examples[sample_id] = {"id": sample_id, "group_id": group_id,
                    "n_positions": 0, "n_values": 0, "squared_error_sum": 0.,
                    "activation_energy_sum": 0., "l0_sum": 0.}
            example = self.examples[sample_id]
            if example["group_id"] != group_id:
                raise ValueError("one example ID cannot belong to different source groups")
            example["n_positions"] += count
            example["n_values"] += count * values.shape[-1]
            example["squared_error_sum"] += error[indices].sum().item()
            example["activation_energy_sum"] += energy[indices].sum().item()
            example["l0_sum"] += l0[indices].sum().item()

    def finish(self):
        if not self.activations.n:
            raise ValueError("evaluation requires at least one selected activation position")
        n = self.activations.n
        tss = (self.reference_tss if self.reference_mean is not None else
               self.activations.m2.sum().item())
        summary = {"n_positions": n, "n_examples": len(self.examples),
            "d_in": self.activations.mean.numel(), "d_sae": self.features.mean.numel(),
            "mse": self.sse / self.n_values, "squared_error_sum": self.sse,
            "variance_denominator_sum": tss, "fvu": self.sse / tss if tss > 0 else None,
            "fvu_defined": tss > 0, "l0": self.l0 / n,
            "fraction_inactive": self.firing_count.eq(0).double().mean().item(),
            "normalized_mse": self.sse / self.energy if self.energy > 0 else None}
        for example in self.examples.values():
            example["mse"] = example["squared_error_sum"] / example["n_values"]
            example["l0"] = example["l0_sum"] / example["n_positions"]
        statistics = {"frequency": self.firing_count.double() / n,
            "firing_count": self.firing_count, "mean": self.features.mean,
            "variance": self.features.m2 / n, "min": self.minimum, "max": self.maximum}
        return summary, list(self.examples.values()), statistics
