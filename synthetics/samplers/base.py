#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass


_T = TypeVar("_T")


@dataclass(kw_only=True, frozen=True)
class DemographicModelInfo(Generic[_T]):
    """Information about a given demographic model"""

    name: str
    path: Path

    def instantiate(self) -> _T:
        """Instantiate underlying model"""
        pass
