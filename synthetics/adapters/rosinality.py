#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

"""
Adapter models for Rosinality stylegan2 reimplementation
"""

from typing import Literal
from pathlib import Path
import math
import torch as pt
import torch.nn as nn
import synthetics.tools.modules as stm
import synthetics.tools.context_manager as stcm
import synthetics.tools.downloads as std


FFHQ_1024_URL="https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing:stylegan2-ffhq-config-f.pt"
GeneratorVariant = Literal["no-ema", "ema"]


_root = Path(__file__).parent.parent
base = stm.LazyLoader(
    "base",
    globals(),
    "model",
    sys_path=(_root/"external/restyle/models/stylegan2").as_posix(),
    )


def _unsqueeze_to(
    target: pt.Tensor, source: pt.Tensor, dim: int = 0) -> pt.Tensor:
    while source.ndim != target.ndim:
        source = source.unsqueeze(dim)
    return source


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


class MappingNetworkAdapter(nn.Module):
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
        activation: str = "fused_lrelu",
        lr_multiplier: float = 0.01,
        w_avg_beta: float = 0.998,
        w_avg_approx_batch_size: int = 16,
        w_avg_approx_samples: int = 2000,
    ) -> None:
        # Sanity check
        if c_dim > 0 or embed_features is not None:
            raise ValueError("Class conditioning is not supported")
        if activation != "fused_lrelu":
            raise ValueError("Only fused_lrelu activation function is supported")
        if layer_features is not None:
            raise ValueError("Intermediate layer dimensions is not supported")
        if z_dim != w_dim:
            raise ValueError("Z and W dimensions must match!")
        # Create layer
        super().__init__()

        # Mapping network
        layers = [base.PixelNorm()]
        for _ in range(num_layers):
            layers.append(base.EqualLinear(
                    w_dim, w_dim, lr_mul=lr_multiplier, activation=activation
                ))
        self.style = nn.Sequential(*layers)

        # Properties
        self.w_dim = w_dim
        self.z_dim = z_dim
        self.num_ws = num_ws
        self._w_avg: pt.Tensor
        self.register_buffer('_w_avg', pt.zeros([w_dim]), persistent=True)
        self._w_avg_need_approximation: bool = True
        self._w_avg_approx_batch_size: int = w_avg_approx_batch_size
        self._w_avg_approx_samples: int = w_avg_approx_samples

    @property
    def w_avg(self) -> pt.Tensor:
        """Average W"""
        if self._w_avg_need_approximation:
            # Funky way to get current device
            device = next(self.parameters()).device
            # Approximate w
            # NOTE: Suboptimal, but does not work otheriwse ...
            self._w_avg = _estimate_w_avg_original(
                self,
                batch_size=self._w_avg_approx_batch_size,
                device=device,
                n_samples=self._w_avg_approx_samples)

            self._w_avg_need_approximation = False
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
        w = self.style(z)
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


class SynthesisNetworkAdapter(nn.Module):
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
        **block_kwargs
    ) -> None:
        super().__init__()

        blur_kernel = [1, 3, 3, 1]
        channel_multiplier = 2
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.log_size = int(math.log(img_resolution, 2))
        self.n_noise_layers = (self.log_size - 2) * 2 + 1

        # Synthetis block
        self.input = base.ConstantInput(self.channels[4])
        self.conv1 = base.StyledConv(
            self.channels[4],
            self.channels[4],
            3,
            w_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb1 = base.ToRGB(
            self.channels[4],
            w_dim,
            upsample=False)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]
        for layer_idx in range(self.n_noise_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', pt.randn(*shape))
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                base.StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    w_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                base.StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    w_dim,
                    blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(base.ToRGB(out_channel, w_dim))
            in_channel = out_channel
        self.num_layers = self.log_size * 2 - 2

    def forward(self, ws: pt.Tensor, **block_kwargs) -> pt.Tensor:
        """Synthesis images"""
        # Noise
        noise = [getattr(self.noises, f'noise_{i}')
                 for i in range(self.n_noise_layers)]

        # Synthesis - 0
        latent = ws
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        # Synthesis > 0
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2],
                self.convs[1::2],
                noise[1::2],
                noise[2::2],
                self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        # Done
        image = skip
        return image


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

    @classmethod
    def download(cls, url: str | None = None) -> Path:
        """Download models"""
        if url is None:
            url = FFHQ_1024_URL
        return std.gdrive_download(url=url)

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
            img_resolution=1024,
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
        self.output = (nn.Identity() if img_resolution == 1024 else
                       nn.AdaptiveAvgPool2d(img_resolution))

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
            # Rename keys accordingly for mapping/synthesis networks
            to_rmv = list(state_dict.keys())
            base = "g_ema" if self.variant == "ema" else "g"
            # W_avg
            if "latent_avg" in state_dict:
                state_dict["mapping._w_avg"] = state_dict["latent_avg"]
            key: str
            for key, value in state_dict[base].items():
                _prefix = "mapping." if key.startswith("style") else "synthesis."
                state_dict[_prefix + key] = value
            for rm in to_rmv:
                state_dict.pop(rm)
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
        gen_output = self.synthesis(
            ws,
            update_emas=update_emas,
            **synthesis_kwargs)
        return self.output(gen_output)
