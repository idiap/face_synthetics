#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from collections.abc import Iterable
from typing import Any
from types import ModuleType
import sys
from importlib import import_module


class LazyLoader(ModuleType):
    """Lazy module loader

    See:
      - https://github.com/tensorflow/agents/blob/v0.19.0/tf_agents/utils/lazy_loader.py
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        name: str,
        sys_path: str | Iterable[str] | None = None
    ) -> None:
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._sys_path = sys_path
        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Insert to sys.path if badly packaged modules
        if self._sys_path is not None:
            if isinstance(self._sys_path, str):
                self._sys_path = [self._sys_path]
            for path in self._sys_path:
                if path not in sys.path:
                    sys.path.insert(0, path)

        # Import the target module and insert it into the parent's namespace
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on
        #   lookups that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item: str):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
