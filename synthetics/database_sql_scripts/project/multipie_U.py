#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os
from bob.bio.face.database import MultipieBioDatabase

database = MultipieBioDatabase(
        original_directory=rc['bob.db.multipie.directory'],
        protocol='U',
        training_depends_on_protocol=True,
    )

seed=0
num_steps=1000
