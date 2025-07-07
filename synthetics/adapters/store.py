#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch as pt


_storage: dict[str, "pt.Tensor"] = {}
_enabled: bool = True


def enable_cache():
    """Enable caching"""
    global _enabled
    _enabled = True


def disable_cache():
    """Disable caching"""
    global _enabled
    _enabled = False


def cache(key: str, value: "pt.Tensor") -> None:
    """Cache a given value"""
    if not _enabled:
        return

    if key in _storage:
        raise KeyError(f"`{key}` already exists!")
    _storage[key] = value


def get(key: str) -> "pt.Tensor":
    """Get value"""
    if not _enabled:
        raise ValueError("Cache is disabled!")
    return _storage[key]
