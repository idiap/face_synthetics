#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from collections.abc import Callable
from typing import Any
import functools

# Hydra-like instantiation from config




def _call_target(
    target_fn: Callable[..., Any],
    partial: bool,
    kwargs: dict[str, Any],
) -> Any:
    if partial:
        try:
            return functools.partial(target_fn, **kwargs)
        except Exception as exc:
            m = (f"Error while creating partial({target_fn.__qualname__}) "
                 f"object: {exc}")
            raise ValueError(m) from exc
    else:
        try:
            return target_fn(**kwargs)
        except Exception as exc:
            m = (f"Error while creating {target_fn.__qualname__} "
                 f"object: {exc}")
            raise ValueError(m) from exc


def _import(module: str) -> Any:
    """Import module from dot path"""
    from importlib import import_module
    from types import ModuleType

    parts = module.split(".")
    if len(parts) == 0:
        raise ValueError(f"Invalid module input: `{module}`")

    # Import base module
    try:
        obj = import_module(parts[0])
    except Exception as exc_import:
        m = f"Error importing {parts[0]}: `{exc_import}`. Is {module} installed?"
        raise ImportError(m) from exc_import

    # Go through childs
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            # Attributes
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{module}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{module}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def _is_target(config: dict[str, Any]) -> bool:
    return "_target_" in config


def _resolve_target(
    target: str | type | Callable[..., Any]
) -> type | Callable[..., Any]:
    """Convert dot path into callable"""
    if isinstance(target, str):
        target = _import(target)
    if not callable(target):
        m = (f"Expected a callable target, got '{target}' of type "
             f"`{type(target).__name__}`")
        raise ValueError(m)
    return target


def _instantiate(config: Any) -> Any:

    if isinstance(config, dict):
        if _is_target(config):
            kwargs = {}
            target_fn = _resolve_target(target=config.get("_target_"))
            partial = config.get("_partial_", False)
            for key, value in config.items():
                if key not in ("_target_", "_partial_"):
                    value = _instantiate(config=value)
                    kwargs[key] = value
            return _call_target(target_fn, partial, kwargs)
        else:
            # Not a target -> return dict
            dict_items = {}
            for key, value in config.items():
                # list items inherits recursive flag from the containing dict.
                dict_items[key] = _instantiate(value)
            return dict_items
    elif isinstance(config, list):
        # List of elements
        items = [_instantiate(element) for element in config]
        return items
    else:
        return config


def instantiate(config: Any) -> Any:
    """Instantiate the object parameterised by a given configuration"""
    if config is None:
        return None

    if isinstance(config, list | dict):
        return _instantiate(config)

    # Unsupported type
    m = (f"Cannot instantiate config of type {type(config).__name__}. Top level"
         " config must be a plain dict/list")
    raise ValueError(m)
