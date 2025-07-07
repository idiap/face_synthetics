#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from pathlib import Path
import torch as pt
import gdown


def gdrive_download(url: str, subfolder: str | Path | None = None) -> Path:
    """
    Download file from a given google drive
    format: <url>:<filename>
    """
    # Resolve url
    url, filename = url.rsplit(":", maxsplit=1)

    # Folder
    model_dir = Path(pt.hub.get_dir(), "checkpoints")
    if subfolder is not None:
        if isinstance(subfolder, str):
            subfolder = Path(subfolder)
        model_dir = model_dir / subfolder
    model_dir.mkdir(parents=True, exist_ok=True)
    cached_file = Path(model_dir, filename)
    if not cached_file.exists():
        gdown.download(
            url=url,
            output=cached_file.as_posix(),
            quiet=False,
            fuzzy=True,
            use_cookies=False,
        )
    return cached_file
