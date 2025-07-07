#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os
from bob.extension import rc
from bob.bio.base.database import FileListBioDatabase

database = FileListBioDatabase(
    filelists_directory=os.path.join(utils.source_path,
                                     'synthetics/config/project/protocols'),
    name='multipie',
    protocol='P_center_lit',
    original_directory=rc['bob.db.multipie.directory'],
    original_extension='.png'
)

seed=0
num_steps=1000