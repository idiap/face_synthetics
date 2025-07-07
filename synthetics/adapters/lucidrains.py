#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

"""
Adapter models for Lucidrains stylegan2 reimplementation
"""
from collections.abc import Iterable
from typing import Literal
import math
import torch as pt
import torch.nn as nn
import stylegan2_pytorch.stylegan2_pytorch as sg2
import synthetics.tools.context_manager as stcm

GeneratorVariant = Literal["no-ema", "ema"]


@pt.no_grad()
def _estimate_w_avg(
    model: "MappingNetworkAdapter",
    batch_size: int,
    device: pt.device,
    n_samples: int
) -> pt.Tensor:
    """Estimage average W"""
    with stcm.evaluating(model):
        n_batch = (n_samples + batch_size - 1) // batch_size
        assert n_batch > 0, "Can not estimate w_avg without any batch!"
        w_avg: pt.Tensor | None = None
        for _ in range(n_batch):
            # Generate noise
            z = pt.randn(batch_size, model.z_dim, device=device)
            # Passthrough mapping network to approximage `w_avg`
            w: pt.Tensor = model(z, estimate_w_avg=True)
            partial_w_sum = w.sum(dim=0, keepdim=True)
            if w_avg is None:
                w_avg = partial_w_sum
            else:
                w_avg += partial_w_sum
        # Compute mean
        w_avg = w_avg.div(float(n_batch * batch_size))
        return w_avg


@pt.no_grad()
def _estimate_w_avg_original(
    model: "MappingNetworkAdapter",
    batch_size: int,
    device: pt.device,
    n_samples: int
) -> pt.Tensor:
    """Estimage average W"""
    with stcm.evaluating(model):
        ws: list[pt.Tensor] = []
        z = pt.randn(n_samples, model.z_dim, device=device)
        n_chunk = (n_samples + batch_size - 1) // batch_size
        for chunk in z.chunk(n_chunk, dim=0):
            # Passthrough mapping network to approximage `w_avg`
            w: pt.Tensor = model(chunk, estimate_w_avg=True)
            ws.append(w)
        # Compute mean
        w_avg = pt.cat(ws, dim=0)
        return w_avg.mean(dim=0)    # [w_dim]


def _unsqueeze_to(
    target: pt.Tensor, source: pt.Tensor, dim: int = 0) -> pt.Tensor:
    while source.ndim != target.ndim:
        source = source.unsqueeze(dim)
    return source


def _styles_def_to_tensor(
    styles_def: Iterable[tuple[pt.Tensor, int]]
) -> pt.Tensor:
    """Convert styles"""
    return pt.cat(
        [tensor[:, None, :].expand(-1, mult, -1)
         for tensor, mult in styles_def],
        dim=1)


class MappingNetworkAdapter(sg2.StyleVectorizer):
    """
    Mapping Network following Nvidia interface

    Args:
        z_dim: Input latent (Z) dimensionality.
        c_dim: Conditioning label (C) dimensionality, 0 = no label (not used)
        w_dim: Intermediate latent (W) dimensionality
        num_ws: Number of intermediate latents to output, None = do not broadcast.
        num_layers: Number of mapping layers. Defaults to 8.
        embed_features: Label embedding dimensionality, None = same as w_dim. Defaults to None.
        layer_features: Number of intermediate features in the mapping layers, None = same as w_dim. Defaults to None.
        activation: Activation function. Defaults to "lrelu".
        lr_multiplier: Learning rate multiplier for the mapping layers. Defaults to 0.1.
        w_avg_beta: Decay for tracking the moving average of W during training, None = do not track. Defaults to 0.998.
    """

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        num_ws: int | None,
        num_layers: int = 8,
        embed_features: int | None = None,
        layer_features: int | None = None,
        activation: str = "lrelu",
        lr_multiplier: float = 0.1,
        w_avg_beta: float = 0.998,
        w_avg_approx_batch_size: int = 16,
        w_avg_approx_samples: int = 2000,
    ) -> None:
        # Sanity check
        if c_dim > 0 or embed_features is not None:
            raise ValueError("Class conditioning is not supported")
        if activation != "lrelu":
            raise ValueError("Only leaky relu activation function is supported")
        if layer_features is not None:
            raise ValueError("Intermediate layer dimensions is not supported")
        if z_dim != w_dim:
            raise ValueError("Z and W dimensions must match!")
        # Create layer
        super().__init__(emb=z_dim, depth=num_layers, lr_mul=lr_multiplier)
        self.w_dim = w_dim
        self.z_dim = z_dim
        self.num_ws = num_ws
        self._w_avg: pt.Tensor
        self.register_buffer('_w_avg', pt.zeros([w_dim]), persistent=False)
        self._w_avg_approximated: bool = False
        self._w_avg_approx_batch_size: int = w_avg_approx_batch_size
        self._w_avg_approx_samples: int = w_avg_approx_samples

    @property
    def w_avg(self) -> pt.Tensor:
        """Average W"""
        if not self._w_avg_approximated:
            # Funky way to get current device
            device = next(self.parameters()).device
            # Approximate w
            # NOTE: Suboptimal, but does not work otheriwse ...
            self._w_avg = _estimate_w_avg_original(
                self,
                batch_size=self._w_avg_approx_batch_size,
                device=device,
                n_samples=self._w_avg_approx_samples)
            self._w_avg_approximated = True
        return self._w_avg

    def forward(
        self,
        z: pt.Tensor,
        c: pt.Tensor | None = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: int | None = None,
        update_emas: bool = False,
        estimate_w_avg: bool = False,
    ) -> pt.Tensor:
        """Map Z to W space"""
        self._w_avg_approx_batch_size = z.shape[0]
        # Sanity check
        if c is not None:
            raise ValueError("Class conditioning is not supported!")
        # Map
        w: pt.Tensor = super().forward(z)
        if estimate_w_avg:
            return w

        # Broadcast?
        if self.num_ws is not None:
            w = w.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Truncate if needed
        if truncation_psi != 1.0:
            w_avg = _unsqueeze_to(w, self.w_avg, dim=0)
            if self.num_ws is None or truncation_cutoff is None:
                w = w_avg.lerp(w, truncation_psi)
            else:
                raise RuntimeError("Truncation is not supporte yet!")
        # Done
        return w


class SynthesisNetworkAdapter(sg2.Generator):
    """
    Synthesis Network following Nvidia interface

    Args:
        w_dim: Intermediate latent (W) dimensionality
        img_resolution: Output image resolution
        img_channels: Number of color channels **N/A**
        channel_base: Overall multiplier for the number of channels. Defaults to 32768.
        channel_max: Maximum number of channels in any layer. Defaults to 512.
        num_fp16_res: Use FP16 for the N highest resolutions. Defaults to 4. **N/A**
        block_kwargs: Arguments for SynthesisBlock. **N/A**
    """

    def __init__(
        self,
        w_dim: int,
        img_resolution: int,
        img_channels: int,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_fp16_res: int = 4,
        remap_output: bool = False,
        **block_kwargs
    ) -> None:
        super().__init__(
            image_size=img_resolution,
            latent_dim=w_dim,
            network_capacity=16,
            transparent=False,
            attn_layers=[],
            no_const=False,
            fmap_max=channel_max,
        )
        self.image_size = img_resolution
        self.num_layers = int(math.log2(img_resolution) - 1)
        self.remap_output = remap_output

    def forward(self, ws: pt.Tensor, **block_kwargs) -> pt.Tensor:
        """Synthesis images"""
        batch_size = ws.shape[0]
        # Make Ws -> different from orginal SG
        # _ws = _styles_def_to_tensor([(ws, self.num_layers)])
        noise = (pt.FloatTensor(
            batch_size,
            self.image_size,
            self.image_size,
            1)
                 .to(ws.device)
                 .uniform_(0.0, 1.0))
        # Synthesis
        assert ws.shape[1] == self.num_layers
        images: pt.Tensor = super().forward(ws, noise)
        images = images.clamp(0.0, 1.0)
        # Map to [-1, 1]
        if self.remap_output:
            images = (images * 2.0) - 1.0
        return images


class GeneratorAdapter(nn.Module):
    """
    Generator Network following Nvidia interface

    Args:
        z_dim: Input latent (Z) dimensionality.
        c_dim: Conditioning label (C) dimensionality, 0 = no label (not used)
        w_dim: Intermediate latent (W) dimensionality
        img_resolution: Output resolution
        img_channels: Number of output color channels
        mapping_kwargs: Arguments for MappingNetwork
        synthetis_kwargs: Arguments for SynthesisNetwork
    """

    _version = 2

    def __init__(self,
        z_dim,                        # Input latent (Z) dimensionality.
        c_dim,                        # Conditioning label (C) dimensionality.
        w_dim,                        # Intermediate latent (W) dimensionality.
        img_resolution,               # Output resolution.
        img_channels,                 # Number of output color channels.
        mapping_kwargs      = None,   # Arguments for MappingNetwork.
        synthesis_kwargs    = None,   # Arguments for SynthesisNetwork.
        variant: GeneratorVariant = "ema",   # Which version to restore
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.variant = variant
        self.synthesis = SynthesisNetworkAdapter(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **(synthesis_kwargs or {}),
        )
        self.num_ws = self.synthesis.num_layers
        self.mapping = MappingNetworkAdapter(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.num_ws,
            **(mapping_kwargs or {}),
            )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str]
    ) -> None:

        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Update
            if "GAN" not in state_dict:
                raise ValueError(
                    "Version 1 is expecting a `GAN` entry in its state_dict, "
                    "got None!"
                    )
            load_ema = self.variant == "ema"
            # Mapping
            _mapping_key = "SE" if load_ema else "S"
            _synth_key = "GE" if load_ema else "G"
            key: str
            for key, value in state_dict["GAN"].items():
                if key.startswith(_mapping_key):
                    _key = "mapping" + key.removeprefix(_mapping_key)
                    state_dict[_key] = value
                elif key.startswith(_synth_key):
                    _key = "synthesis" + key.removeprefix(_synth_key)
                    state_dict[_key] = value
            # Remove unused keys
            state_dict.pop("GAN")
            state_dict.pop("version")
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)

    def forward(
        self,
        z: pt.Tensor,
        c: pt.Tensor | None = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: int | None = None,
        update_emas: bool = False,
        **synthesis_kwargs
    ) -> pt.Tensor:
        """Generate images"""
        # Z -> Ws
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
            )
        # Synthesis
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

