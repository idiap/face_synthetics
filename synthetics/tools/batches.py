#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from collections.abc import Iterator
import torch as pt
import numpy as np


def make_batch(
    array: np.ndarray | pt.Tensor,
    size: int
) -> Iterator[np.ndarray | pt.Tensor]:
    """Convert tensor to batches"""
    n_batch = (array.shape[0] + size - 1) // size
    for k in range(n_batch):
        start = k * size
        stop = min((k + 1) * size, array.shape[0])
        yield array[start:stop]
