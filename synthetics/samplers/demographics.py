#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from collections.abc import Iterable
from typing import TYPE_CHECKING
import joblib
import numpy as np
import synthetics.samplers.base as ssb


if TYPE_CHECKING:
    from sklearn.mixture._base import BaseMixture


class MixtureInfo(ssb.DemographicModelInfo["BaseMixture"]):
    """Sklearn-based density model"""

    def instantiate(self) -> "BaseMixture":
        return joblib.load(self.path)


class MixtureSampler:
    """Mixture-based sampler"""

    def __init__(
        self, densities: Iterable[MixtureInfo],
        dtype: np.dtype = np.float32,
    ) -> None:
        self.densities = {d.name: d.instantiate() for d in densities}
        self.dtype = dtype

    def sample(
        self, n_samples: int | dict[str, int]
    ) -> tuple[np.ndarray, Iterable[str]]:
        """Sample from the underlying mixture"""

        # Format
        if isinstance(n_samples, int):
            n_samples = {name: n_samples for name, _ in self.densities.items()}

        # Sample accordingly
        samples: list[np.ndarray] = []
        labels: list[str] = []
        for name, n_sample in n_samples.items():
            density = self.densities[name]
            sample, _ = density.sample(n_samples=n_sample)
            samples.append(sample.astype(self.dtype))
            labels.extend([name] * n_sample)
        return np.concatenate(samples, axis=0), labels
