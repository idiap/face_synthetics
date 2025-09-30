#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from collections.abc import Sequence, Mapping
from typing import cast, Any, TypeAlias
import torch as pt
import torch.nn as nn
import synthetics.pytorch.activations as spa


Kwargs: TypeAlias = Mapping[str, Any]


class StackedLinear(nn.Sequential):
    """Sequence of stacked Linear + Activation"""

    def __init__(
        self,
        in_channels: int,
        sizes: list[int],
        activations: str | Sequence[str],
        activation_kwargs: Kwargs | Sequence[Kwargs] | None = None,
    ):
        # Sanity check
        if isinstance(activations, str):
            activations = [activations]
        if len(activations) == 1 and len(sizes) > 1:
            activations = cast(list[str], activations * len(sizes))  # type: ignore [operator]
        if len(activations) != len(sizes):
            m = (
                "Arguments `sizes` and `activations` must have the same size, "
                f"got {len(sizes)} != {len(activations)}"
            )
            raise ValueError(m)
        if activation_kwargs is None:
            activation_kwargs = {}
        if isinstance(activation_kwargs, Mapping):
            activation_kwargs = [activation_kwargs]
        if len(activation_kwargs) == 1 and len(sizes) > 1:
            activation_kwargs = cast(
                list[Kwargs],
                activation_kwargs * len(sizes))  # type: ignore [operator]
        if len(activation_kwargs) != len(sizes):
            m = (
                "Arguments `sizes` and `activations_kwargs` must have the same "
                f"size, got {len(sizes)} != {len(activation_kwargs)}"
            )
            raise ValueError(m)

        # Build
        layers: list[nn.Module] = []
        _in_channels = [in_channels] + sizes[:-1]
        for c_in, c_out, act_fn in zip(_in_channels, sizes, activations):
            lin = nn.Linear(in_features=c_in, out_features=c_out, bias=True)
            act = spa.by_name(act_fn)()
            layers.append(lin)
            layers.append(act)
        super().__init__(*layers)


class LinearAutoEncoder(nn.Module):
    """Symetric Linear autoencoder"""

    def __init__(
        self,
        in_channels: int,
        sizes: list[int],
        activations: str | Sequence[str],
        activation_kwargs: Kwargs | Sequence[Kwargs] | None = None,
    ) -> None:
        super().__init__()
        # Encoder -> Compress
        self.encoder = StackedLinear(
            in_channels=in_channels,
            sizes=sizes,
            activations=activations,
        )
        # Decoder -> Reconstruct
        if not isinstance(activations, str) and isinstance(activations, Sequence):
            activations = list(reversed(activations))
        if isinstance(activation_kwargs, Sequence):
            activation_kwargs = list(reversed(activation_kwargs))
        self.decoder = StackedLinear(
            in_channels=sizes[-1],
            sizes=list(reversed(sizes[:-1])) + [in_channels],
            activations=activations,
            activation_kwargs=activation_kwargs,
        )

    def forward(self, x: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
        """Encode + Decode inputs"""
        x_embed = self.encoder(x)
        x_rec = self.decoder(x_embed)
        return x_rec, x_embed