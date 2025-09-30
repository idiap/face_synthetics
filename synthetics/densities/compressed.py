#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TYPE_CHECKING
from collections.abc import Iterable
from pathlib import Path
import torch as pt
import synthetics.tools.batches as stb


if TYPE_CHECKING:
    import synthetics.pytorch.models.autoencoders as spm
    import synthetics.samplers.demographics as ssd


def _load_ckpt(filename: str | Path, prefix: str | None):
    raw: dict[str, pt.Tensor] = pt.load(filename, map_location="cpu")
    ckpt = {}
    for key, value in raw.items():
        if prefix is not None:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
            else:
                continue
        ckpt[key] = value
    del raw
    return ckpt


class CompressedGMMs:
    """ GMMs + decoder """

    def __init__(
        self,
        w_dim: int,     # If positive, assumes to be generating Wplus
        sampler: "ssd.MixtureSampler",
        autoencoder: "spm.LinearAutoEncoder",
        filename: str | Path | None = None,
        ckpt_prefix: str | None = None,
        device: str = "cuda",
    ) -> None:

        # Sampler
        self.sampler = sampler

        # Init decoder
        decoder = autoencoder.decoder
        if filename is not None:
            ckpt = _load_ckpt(filename, ckpt_prefix)
            decoder.load_state_dict(ckpt)
        self.decoder = decoder.eval().to(device=device)

        # Properties
        self.device = device
        self.w_dim = w_dim

    @pt.inference_mode()
    def __call__(
        self,
        n_samples: int | dict[str, int],
        batch_size: int = 512) -> tuple[pt.Tensor, Iterable[str]]:
        # Sample W subspace
        w_subspace, labels = self.sampler.sample(n_samples=n_samples)   # [N, 512]

        # Decode to W space or W+
        ws = []
        for batch in stb.make_batch(w_subspace, batch_size):
            inputs = pt.from_numpy(batch).to(self.device)
            w: pt.Tensor = self.decoder(inputs)
            if self.w_dim > 0:
                w = w.reshape(w.shape[0], -1, self.w_dim)
            ws.append(w)
        return pt.concat(ws, 0), labels
