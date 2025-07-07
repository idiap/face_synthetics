#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TYPE_CHECKING
from contextlib import contextmanager


if TYPE_CHECKING:
    import torch.nn as nn


@contextmanager
def evaluating(net: "nn.Module"):
    """
    Temporarily switch to evaluation mode.
    See: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/2
    """
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()