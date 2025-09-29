#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TYPE_CHECKING
from collections.abc import Callable
from torch.nn.modules import activation


if TYPE_CHECKING:
    import torch.nn as nn


Activation = Callable[..., "nn.Module"]
_pt_activations = {
    str(a).lower(): getattr(activation, a) for a in activation.__all__
}


def by_name(act: str) -> Activation:
    """Get activation class by name"""
    if (act := act.lower()) in _pt_activations:
        return _pt_activations[act]
    # Unknown Activation
    raise KeyError(f"Cannot find activation function for string <{act}>")
